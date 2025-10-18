#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rgb_to_motionbert3d_patched_v5.py (explicit-config, no autodetect, debug-friendly)

- Reads three explicit JSON configs: YOLO, RTMPose, MotionBERT.
- No auto-detection. Every important behavior comes from cfg or CLI.
- CLI overrides config; config overrides built-in defaults.
- Adds --debug diagnostics and a config checklist dump.

Artifacts (in --debug_dir):
  - io_introspect.txt
  - model_io.json
  - config_resolved.json
  - config_checklist.json
  - yolo_decode.txt
  - detections.jsonl
  - coco_2d.npy, h36m_2d.npy, mb_input_seq.npy, mb_output_3d.npy
  - overlay.mp4 (when --save_overlay)
"""

import argparse
import json
import os
import sys
import time
import platform
from typing import Tuple, List
import numpy as np
import cv2
import onnxruntime as ort

# ---------------------------
# Utils
# ---------------------------

def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    shape = img.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(x):
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(boxes, scores, iou_thr=0.45, max_det=300):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def crop_to_aspect(img, box_xyxy, out_hw=(256,192), scale=1.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    out_h, out_w = out_hw
    target_ar = out_w / out_h  # 192/256
    if bw / bh > target_ar:
        bh = bw / target_ar
    else:
        bw = bh * target_ar

    x1n = max(0, int(round(cx - bw / 2)))
    y1n = max(0, int(round(cy - bh / 2)))
    x2n = min(w - 1, int(round(cx + bw / 2)))
    y2n = min(h - 1, int(round(cy + bh / 2)))

    crop = img[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        return None, None
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    rect = (x1n, y1n, x2n - x1n, y2n - y1n)
    return resized, rect

def simcc_decode(simcc_x, simcc_y, split_ratio=2.0):
    def softmax(a, axis=-1):
        a = a - np.max(a, axis=axis, keepdims=True)
        ea = np.exp(a)
        return ea / np.sum(ea, axis=axis, keepdims=True)
    px = softmax(simcc_x, axis=2)
    py = softmax(simcc_y, axis=2)
    x_idx = np.argmax(px, axis=2)
    y_idx = np.argmax(py, axis=2)
    x = x_idx.astype(np.float32) / float(split_ratio)
    y = y_idx.astype(np.float32) / float(split_ratio)
    conf = np.sqrt(np.max(px, axis=2) * np.max(py, axis=2)).astype(np.float32)
    coords = np.stack([x, y], axis=-1)
    return coords, conf

def coco17_to_h36m17(coco_xy, coco_conf):
    nose, leye, reye = coco_xy[0], coco_xy[1], coco_xy[2]
    lsho, rsho = coco_xy[5], coco_xy[6]
    lelb, relb = coco_xy[7], coco_xy[8]
    lwri, rwri = coco_xy[9], coco_xy[10]
    lhip, rhip = coco_xy[11], coco_xy[12]
    lknee, rknee = coco_xy[13], coco_xy[14]
    lank, rank = coco_xy[15], coco_xy[16]

    pelvis = (lhip + rhip) / 2.0
    neck   = (lsho + rsho) / 2.0
    spine1 = (pelvis + neck) / 2.0
    eyes_mid = (leye + reye) / 2.0 if (leye.any() and reye.any()) else nose
    head = eyes_mid
    site = nose

    h36m = np.zeros((17, 2), dtype=np.float32)
    h36m[0]  = pelvis
    h36m[1]  = rhip;   h36m[2]  = rknee;  h36m[3]  = rank
    h36m[4]  = lhip;   h36m[5]  = lknee;  h36m[6]  = lank
    h36m[7]  = spine1; h36m[8]  = neck;   h36m[9]  = head; h36m[10] = site
    h36m[11] = lsho;   h36m[12] = lelb;   h36m[13] = lwri
    h36m[14] = rsho;   h36m[15] = relb;   h36m[16] = rwri

    c = coco_conf
    h36m_conf = np.zeros((17,), dtype=np.float32)
    c_pelvis = (c[11] + c[12]) / 2
    c_neck   = (c[5] + c[6]) / 2
    c_spine1 = (c_pelvis + c_neck) / 2
    c_head   = (c[1] + c[2]) / 2 if (c[1] > 0 and c[2] > 0) else c[0]
    c_site   = c[0]
    h36m_conf[0]  = c_pelvis
    h36m_conf[1]  = c[12]; h36m_conf[2]  = c[14]; h36m_conf[3]  = c[16]
    h36m_conf[4]  = c[11]; h36m_conf[5]  = c[13]; h36m_conf[6]  = c[15]
    h36m_conf[7]  = c_spine1; h36m_conf[8] = c_neck; h36m_conf[9] = c_head; h36m_conf[10] = c_site
    h36m_conf[11] = c[5]; h36m_conf[12] = c[7]; h36m_conf[13] = c[9]
    h36m_conf[14] = c[6]; h36m_conf[15] = c[8]; h36m_conf[16] = c[10]
    return h36m, h36m_conf

def normalize_to_minus1_1(xy, img_w, img_h):
    s = min(img_w, img_h) / 2.0
    cx, cy = img_w / 2.0, img_h / 2.0
    xy_norm = xy.copy()
    xy_norm[..., 0] = (xy[..., 0] - cx) / s
    xy_norm[..., 1] = (xy[..., 1] - cy) / s
    return xy_norm

def draw_h36m_skeleton(im, pts, color=(0,255,0), radius=3, thickness=2):
    pairs = [
        (0,1),(1,2),(2,3),
        (0,4),(4,5),(5,6),
        (0,7),(7,8),(8,9),(9,10),
        (8,11),(11,12),(12,13),
        (8,14),(14,15),(15,16)
    ]
    for i,j in pairs:
        p1 = tuple(np.round(pts[i]).astype(int))
        p2 = tuple(np.round(pts[j]).astype(int))
        cv2.line(im, p1, p2, color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(im, tuple(np.round(p).astype(int)), radius, (0,0,255), -1)
    return im

# ---------------------------
# Models
# ---------------------------

class YOLOv8ONNX:
    def __init__(self, model_path, providers=None, conf_thres=0.25, iou_thres=0.45, debug_file=None,
                 output_units='normalized', output_coords='letterbox', person_class_id=0, class_activation='sigmoid',
                 letterbox_target=(640,640)):
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.debug_file = debug_file
        self.output_units = output_units  # 'normalized' | 'pixels'
        self.output_coords = output_coords  # 'letterbox' | 'original'
        self.person_class_id = person_class_id
        self.class_activation = class_activation
        self.letterbox_target = letterbox_target
        if self.debug_file:
            open(self.debug_file, 'w').close()

    def _write_debug(self, line):
        if self.debug_file:
            with open(self.debug_file, 'a') as f:
                f.write(line + '\n')

    def _decode(self, out, r, pad, orig_shape):
        preds = out[0].T  # [N,84]
        boxes_cxcywh = preds[:, :4]
        cls_logits   = preds[:, 4:]
        cls_probs = sigmoid(cls_logits) if self.class_activation == 'sigmoid' else cls_logits
        cls_ids    = np.argmax(cls_probs, axis=1)
        cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]

        m = (cls_ids == self.person_class_id) & (cls_scores >= self.conf_thres)
        if not np.any(m):
            return None, 0.0, 0

        boxes = boxes_cxcywh[m]
        scores = cls_scores[m]

        # Convert to xyxy in letterbox domain
        boxes_xyxy_lb = xywh2xyxy(boxes)

        # If normalized units, scale by target letterbox size
        if self.output_units == 'normalized':
            lb_w, lb_h = self.letterbox_target[1], self.letterbox_target[0]  # (w,h) from (h,w)
            boxes_xyxy_lb[:, [0,2]] *= float(lb_w)
            boxes_xyxy_lb[:, [1,3]] *= float(lb_h)

        # Map to original pixels
        if self.output_coords == 'original':
            boxes_xyxy = boxes_xyxy_lb.copy()
        else:
            dw, dh = pad
            boxes_xyxy = boxes_xyxy_lb.copy()
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy_lb[:, [0, 2]] - dw) / r
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy_lb[:, [1, 3]] - dh) / r

        h0, w0 = orig_shape[:2]
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0 - 1)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0 - 1)

        keep = nms(boxes_xyxy, scores, self.iou_thres, max_det=50)
        if not keep:
            return None, 0.0, np.sum(m)
        best = keep[0]
        return boxes_xyxy[best].astype(np.float32), float(scores[best]), int(np.sum(m))

    def detect_person(self, im_bgr, t=None):
        lb_h, lb_w = self.letterbox_target[0], self.letterbox_target[1]
        im_letter, r, (dw, dh) = letterbox(im_bgr, (lb_h, lb_w))
        im_rgb = cv2.cvtColor(im_letter, cv2.COLOR_BGR2RGB)
        x = (im_rgb.astype(np.float32) / 255.0).transpose(2,0,1)[None, ...]
        out = self.session.run(None, {self.input_name: x})
        bbox, score, valid = None, 0.0, 0
        try:
            bbox, score, valid = self._decode(out[0], r, (dw, dh), im_bgr.shape)
        except Exception:
            valid = 0
        if self.debug_file is not None and t is not None:
            self._write_debug(f"t={t} valid_person_candidates={valid} chosen_score={score:.3f} bbox={None if bbox is None else [float(f) for f in bbox]}")
        return bbox, score

class RTMPoseONNX:
    def __init__(self, model_path, providers=None, split_ratio=2.0):
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.split_ratio = split_ratio

    def _preproc(self, crop_bgr, mode, mean=None, std=None):
        if mode == 'rgb_ms':
            im = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            mean = np.array(mean or [123.675, 116.28, 103.53], dtype=np.float32)
            std  = np.array(std  or [58.395, 57.12, 57.375], dtype=np.float32)
            im = (im - mean) / std
        elif mode == 'bgr_ms':
            im = crop_bgr.astype(np.float32)
            mean = np.array(mean or [123.675, 116.28, 103.53], dtype=np.float32)
            std  = np.array(std  or [58.395, 57.12, 57.375], dtype=np.float32)
            im = (im - mean) / std
        elif mode == 'rgb_255':
            im = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        elif mode == 'bgr_255':
            im = crop_bgr.astype(np.float32) / 255.0
        else:
            raise SystemExit(f"Invalid rtm_preproc: {mode}")
        x = im.transpose(2,0,1)[None, ...]
        return x

    def infer(self, crop_bgr, mode, mean=None, std=None):
        x = self._preproc(crop_bgr, mode, mean=mean, std=std)
        simcc_x, simcc_y = self.session.run(None, {self.input_name: x})
        coords, conf = simcc_decode(simcc_x, simcc_y, split_ratio=self.split_ratio)
        return coords[0], conf[0]

# ---------------------------
# Config helpers
# ---------------------------

def _load_json_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)

def _apply_cfg_map(args, defaults, cfg, mapping):
    # not used for nested; kept for potential flat overrides
    for ck, ak in mapping.items():
        if ck in cfg and hasattr(args, ak):
            if getattr(args, ak) == defaults.get(ak):
                setattr(args, ak, cfg[ck])

def _dump_resolved(args, debug_dir):
    out = os.path.join(debug_dir, "config_resolved.json")
    data = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return out

def _validate_cfgs(debug_dir, ycfg, rcfg, mcfg, y_inout, r_inout):
    y_in, y_out = y_inout
    r_in, r_out = r_inout
    checklist = {"yolo": {}, "rtmpose": {}, "motionbert": {}}

    # YOLO
    yi = (ycfg.get("model", {}).get("input", {}) if ycfg else {})
    yo = (ycfg.get("model", {}).get("output", {}) if ycfg else {})
    checklist["yolo"]["input.shape_runtime"]  = y_in[0][1]
    checklist["yolo"]["output.shape_runtime"] = y_out[0][1]
    checklist["yolo"]["output.shape_hint"]    = yo.get("shape_hint")
    checklist["yolo"]["domain.units"]         = yo.get("domain", {}).get("units")
    checklist["yolo"]["domain.coords"]        = yo.get("domain", {}).get("coords")
    checklist["yolo"]["class_activation"]     = yo.get("class_score_activation")
    checklist["yolo"]["person_id"]            = yo.get("classes", {}).get("person_id")

    # RTM
    ri = (rcfg.get("model", {}).get("input", {}) if rcfg else {})
    ro = (rcfg.get("model", {}).get("output", {}) if rcfg else {})
    checklist["rtmpose"]["input.shape_runtime"]  = r_in[0][1]
    checklist["rtmpose"]["output.heads"]         = ro.get("heads")
    checklist["rtmpose"]["preprocess.mode"]      = ri.get("preprocess", {}).get("mode")
    checklist["rtmpose"]["simcc.split_ratio"]    = ro.get("simcc", {}).get("split_ratio")

    # MB
    mi = (mcfg.get("model", {}).get("input", {}) if mcfg else {})
    mo = (mcfg.get("model", {}).get("output", {}) if mcfg else {})
    checklist["motionbert"]["input.sequence_length"] = mi.get("sequence_length")
    checklist["motionbert"]["output.root_relative"]  = mo.get("root_relative")
    checklist_path = os.path.join(debug_dir, "config_checklist.json")
    with open(checklist_path, "w") as f:
        json.dump(checklist, f, indent=2)
    return checklist_path

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # config files
    ap.add_argument('--cfg_yolo', type=str)
    ap.add_argument('--cfg_rtm', type=str)
    ap.add_argument('--cfg_mb',  type=str)
    ap.add_argument('--debug', action='store_true')

    # io
    ap.add_argument('--video', type=str, required=True)
    ap.add_argument('--out', type=str, default='out_3d.json')
    ap.add_argument('--debug_dir', type=str, default='debug')
    ap.add_argument('--save_overlay', action='store_true')
    ap.add_argument('--overlay_start', type=int, default=0)

    # models (paths can be filled from cfg)
    ap.add_argument('--yolo', type=str, default=None)
    ap.add_argument('--rtmpose', type=str, default=None)
    ap.add_argument('--motionbert', type=str, default=None)

    # providers override
    ap.add_argument('--providers', type=str, nargs='*', default=None)

    # yolo runtime
    ap.add_argument('--person_conf', type=float, default=0.25)
    ap.add_argument('--person_iou', type=float, default=0.45)

    # rtm runtime
    ap.add_argument('--rtm_preproc', type=str, default='rgb_255', choices=['rgb_ms','bgr_ms','rgb_255','bgr_255'])
    ap.add_argument('--simcc_ratio', type=float, default=2.0)
    ap.add_argument('--dump_crops', type=int, default=0)

    # mb runtime
    ap.add_argument('--wrap_pad', action='store_true')
    ap.add_argument('--rootrel', action='store_true')

    args = ap.parse_args()
    os.makedirs(args.debug_dir, exist_ok=True)
    defaults = {a.dest: a.default for a in ap._actions if a.dest and a.dest != 'help'}

    # Load configs and merge (CLI > cfg > defaults)
    ycfg = {}; rcfg={}; mcfg={}
    if args.cfg_yolo and os.path.isfile(args.cfg_yolo):
        ycfg = _load_json_cfg(args.cfg_yolo)
        y_model = ycfg.get("model", {})
        y_out   = y_model.get("output", {})
        y_post  = ycfg.get("postprocess", {})
        y_in    = y_model.get("input", {})
        if args.yolo in (None, "") and "path" in y_model:
            args.yolo = y_model["path"]
        if args.providers is None and "providers" in y_model:
            args.providers = y_model["providers"]
        if float(args.person_conf) == defaults.get("person_conf") and "conf_threshold" in y_post:
            args.person_conf = float(y_post["conf_threshold"])
        if float(args.person_iou) == defaults.get("person_iou") and "iou_threshold" in y_post:
            args.person_iou = float(y_post["iou_threshold"])
    if args.cfg_rtm and os.path.isfile(args.cfg_rtm):
        rcfg = _load_json_cfg(args.cfg_rtm)
        r_model = rcfg.get("model", {})
        r_in    = r_model.get("input", {})
        r_pre   = r_in.get("preprocess", {})
        r_out   = r_model.get("output", {})
        r_sim   = r_out.get("simcc", {})
        r_run   = rcfg.get("runtime", {})
        if args.rtmpose in (None, "") and "path" in r_model:
            args.rtmpose = r_model["path"]
        if args.providers is None and "providers" in r_model:
            args.providers = r_model["providers"]
        if args.rtm_preproc == defaults.get("rtm_preproc") and "mode" in r_pre:
            args.rtm_preproc = r_pre["mode"]
        if float(args.simcc_ratio) == defaults.get("simcc_ratio") and "split_ratio" in r_sim:
            args.simcc_ratio = float(r_sim["split_ratio"])
        if int(args.dump_crops) == defaults.get("dump_crops") and "dump_first_k_crops" in r_run:
            args.dump_crops = int(r_run["dump_first_k_crops"])
    if args.cfg_mb and os.path.isfile(args.cfg_mb):
        mcfg = _load_json_cfg(args.cfg_mb)
        m_model = mcfg.get("model", {})
        m_in    = m_model.get("input", {})
        m_out   = m_model.get("output", {})
        m_run   = mcfg.get("runtime", {})
        if args.motionbert in (None, "") and "path" in m_model:
            args.motionbert = m_model["path"]
        if args.providers is None and "providers" in m_model:
            args.providers = m_model["providers"]
        if not args.wrap_pad and m_run.get("wrap_pad_sequence") is True:
            args.wrap_pad = True
        if not args.rootrel and m_out.get("root_relative") is True:
            args.rootrel = True

    if args.debug:
        io_txt = os.path.join(args.debug_dir, "io_introspect.txt")
        with open(io_txt, "w") as f:
            f.write("=== Runtime Introspect ===\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"python: {sys.version}\n")
            f.write(f"platform: {platform.platform()}\n")
            f.write(f"numpy: {np.__version__}\n")
            f.write(f"opencv: {cv2.__version__}\n")
            f.write(f"onnxruntime: {ort.__version__}\n")
            f.write(f"available_providers: {ort.get_available_providers()}\n")
            f.write(f"video: {args.video}\n")

    # sanity required model paths after merge
    missing = [k for k in ("yolo","rtmpose","motionbert") if getattr(args, k) in (None, "")]
    if missing:
        raise SystemExit(f"Missing required model path(s) after config merge: {missing}. Provide on CLI or in the related config.")

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Debug: resolved config
    if args.debug:
        resolved_path = _dump_resolved(args, args.debug_dir)
        print("[DEBUG] Wrote debug/io_introspect.txt")
        print(f"[DEBUG] Resolved config: {resolved_path}")

    # Build models
    # YOLO explicit domain/units
    y_person_id = 0
    y_activation='sigmoid'
    y_units='normalized'
    y_coords='letterbox'
    lb_h, lb_w = 640, 640
    if ycfg:
        y_out = ycfg.get("model",{}).get("output",{})
        y_dom = y_out.get("domain",{})
        y_person_id = y_out.get("classes",{}).get("person_id", 0)
        y_activation = y_out.get("class_score_activation","sigmoid")
        y_units = y_dom.get("units","normalized")
        y_coords = y_dom.get("coords","letterbox")
        y_in = ycfg.get("model",{}).get("input",{})
        lb_h = int(y_in.get("target_height", y_in.get("height", 640)))
        lb_w = int(y_in.get("target_width",  y_in.get("width",  640)))

    f_yolo_dbg = os.path.join(args.debug_dir, 'yolo_decode.txt') if args.debug else None
    yolo = YOLOv8ONNX(args.yolo, providers=args.providers, conf_thres=args.person_conf, iou_thres=args.person_iou,
                      debug_file=f_yolo_dbg, output_units=y_units, output_coords=y_coords,
                      person_class_id=y_person_id, class_activation=y_activation,
                      letterbox_target=(lb_h, lb_w))
    rtm  = RTMPoseONNX(args.rtmpose, providers=args.providers, split_ratio=args.simcc_ratio)

    # Log model I/O
    if args.debug:
        def shapes(session):
            ins = [(i.name, i.shape) for i in session.get_inputs()]
            outs= [(o.name, o.shape) for o in session.get_outputs()]
            return ins, outs
        y_in, y_out = shapes(yolo.session)
        r_in, r_out = shapes(rtm.session)
        with open(os.path.join(args.debug_dir, "model_io.json"), "w") as f:
            json.dump({"yolo_inputs": y_in, "yolo_outputs": y_out,
                       "rtmpose_inputs": r_in, "rtmpose_outputs": r_out}, f, indent=2)
        print("[DEBUG] YOLO I/O:", y_in, y_out)
        print("[DEBUG] RTMPose I/O:", r_in, r_out)
        checklist_path = _validate_cfgs(args.debug_dir, ycfg, rcfg, mcfg, (y_in, y_out), (r_in, r_out))
        print(f"[DEBUG] Config checklist: {checklist_path}")

    # Prepare outputs
    os.makedirs(args.debug_dir, exist_ok=True)
    f_det_jsonl = os.path.join(args.debug_dir, 'detections.jsonl')
    f_coco = os.path.join(args.debug_dir, 'coco_2d.npy')
    f_h36m = os.path.join(args.debug_dir, 'h36m_2d.npy')
    f_mb_in = os.path.join(args.debug_dir, 'mb_input_seq.npy')
    f_mb_out = os.path.join(args.debug_dir, 'mb_output_3d.npy')
    f_stats = os.path.join(args.debug_dir, 'quick_stats.json')
    detections_f = open(f_det_jsonl, 'w')

    vw = None
    overlay_path = os.path.join(args.debug_dir, 'overlay.mp4') if args.save_overlay else None

    seq_coco_xyc = []
    seq_h36m_xyc = []
    prev_bbox = None
    seen_person = False

    # NEW: accumulate YOLO outputs for yolo_out.json
    yolo_bboxes_norm = []  # [ [x1_norm, y1_norm, x2_norm, y2_norm], ... ]
    yolo_scores = []       # [ score, ... ]

    dumped = 0
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bbox, score = yolo.detect_person(frame, t=t)
        if bbox is None:
            if prev_bbox is not None:
                det_box = prev_bbox.copy()
            else:
                h, w = frame.shape[:2]
                det_box = np.array([w*0.25, h*0.1, w*0.75, h*0.9], np.float32)
                score = 0.0
        else:
            det_box = bbox
            prev_bbox = bbox.copy()
            seen_person = True

        x1, y1, x2, y2 = det_box.tolist()
        det_line = {"t": int(t),
                    "bbox": [x1/in_w, y1/in_h, x2/in_w, y2/in_h],
                    "score": float(score),
                    "yolo_output_units": y_units,
                    "yolo_output_coords": y_coords}
        detections_f.write(json.dumps(det_line) + "\n")

        # NEW: store for final yolo_out.json
        yolo_bboxes_norm.append(det_line["bbox"])
        yolo_scores.append(det_line["score"])

        crop, rect = crop_to_aspect(frame, det_box, out_hw=(256,192), scale=1.25)
        if crop is None:
            # repeat last or zeros
            if len(seq_coco_xyc) > 0:
                seq_coco_xyc.append(seq_coco_xyc[-1].copy())
                seq_h36m_xyc.append(seq_h36m_xyc[-1].copy())
            else:
                seq_coco_xyc.append(np.zeros((17,3), np.float32))
                seq_h36m_xyc.append(np.zeros((17,3), np.float32))
            t += 1
            continue

        if args.dump_crops and dumped < args.dump_crops:
            cv2.imwrite(os.path.join(args.debug_dir, f"crop_{t:05d}.jpg"), crop)
            dumped += 1

        coords_in, conf = rtm.infer(crop, args.rtm_preproc)

        rx, ry, rw, rh = rect
        x_img = rx + coords_in[:, 0] * (rw / 192.0)
        y_img = ry + coords_in[:, 1] * (rh / 256.0)
        coco_xy = np.stack([x_img, y_img], axis=-1).astype(np.float32)
        coco_conf = conf.astype(np.float32)

        h36m_xy, h36m_conf = coco17_to_h36m17(coco_xy, coco_conf)
        seq_coco_xyc.append(np.concatenate([coco_xy, coco_conf[:,None]], axis=1))
        seq_h36m_xyc.append(np.concatenate([h36m_xy, h36m_conf[:,None]], axis=1))

        if args.save_overlay and seen_person and t >= args.overlay_start:
            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter(overlay_path, fourcc, fps, (in_w, in_h))
            overlay = frame.copy()
            overlay = draw_h36m_skeleton(overlay, h36m_xy)
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
            vw.write(overlay)

        t += 1

    detections_f.close()
    cap.release()
    if vw is not None:
        vw.release()

    seq_coco_xyc = np.array(seq_coco_xyc, dtype=np.float32)
    seq_h36m_xyc = np.array(seq_h36m_xyc, dtype=np.float32)
    stats = {
        "frames": int(seq_coco_xyc.shape[0]),
        "coco_conf_mean": float(seq_coco_xyc[...,2].mean() if seq_coco_xyc.size else 0.0),
        "coco_x_ptp": float(seq_coco_xyc[...,0].ptp() if seq_coco_xyc.size else 0.0),
        "coco_y_ptp": float(seq_coco_xyc[...,1].ptp() if seq_coco_xyc.size else 0.0),
        "seen_person": bool(seen_person),
    }
    np.save(f_coco, seq_coco_xyc)
    np.save(f_h36m, seq_h36m_xyc)

    # MotionBERT
    T = 243
    seq = seq_h36m_xyc.copy()
    if args.wrap_pad:
        if len(seq) < T:
            pad = np.repeat(seq[-1][None, ...], T - len(seq), axis=0) if len(seq) else np.zeros((T,17,3), np.float32)
            seq = np.concatenate([seq, pad], axis=0) if len(seq) else pad
        elif len(seq) > T:
            seq = seq[:T]
    else:
        if len(seq) != T:
            raise SystemExit(f"MotionBERT expects T=243, got {len(seq)}. Use --wrap_pad to auto-pad.")

    xy_norm = normalize_to_minus1_1(seq[..., :2], in_w, in_h)
    mb_in_seq = np.concatenate([xy_norm, seq[..., 2:3]], axis=-1)  # [T,17,3]
    np.save(f_mb_in, mb_in_seq)

    mb_sess = ort.InferenceSession(args.motionbert, providers=args.providers or ['CPUExecutionProvider'])
    mb_in_name = mb_sess.get_inputs()[0].name
    mb_out = mb_sess.run(None, {mb_in_name: mb_in_seq[None, ...].astype(np.float32)})[0]  # [1,T,17,3]
    X3D = mb_out[0]
    if args.rootrel:
        X3D[:, 0, :] = 0.0
    np.save(f_mb_out, X3D)

    # Primary 3D result (unchanged)
    result = {
        "video": os.path.basename(args.video),
        "T": int(T),
        "h36m_order": ["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                       "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
                       "RShoulder","RElbow","RWrist"],
        "coords_3d": X3D.tolist()
    }
    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f"Saved 3D pose to {args.out}")
    print(f"[DEBUG] Wrote debug files to: {args.debug_dir}")

    # ---------------------------
    # NEW: Additional JSON outputs
    # ---------------------------

    # 1) rtm_out.json : raw COCO-17 2D keypoints (x,y,conf) per frame
    coco_order = [
        "Nose","LEye","REye","LEar","REar",
        "LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist",
        "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"
    ]
    rtm_payload = {
        "video": os.path.basename(args.video),
        "frames": int(seq_coco_xyc.shape[0]),
        "coco_order": coco_order,
        "coords_2d": seq_coco_xyc.tolist()  # shape [F,17,3] -> (x,y,conf)
    }
    with open("rtm_out.json", "w") as f:
        json.dump(rtm_payload, f)
    print("Saved 2D keypoints to rtm_out.json")

    # 2) yolo_out.json : per-frame normalized bbox and score
    yolo_payload = {
        "video": os.path.basename(args.video),
        "frames": len(yolo_bboxes_norm),
        "bboxes_norm": yolo_bboxes_norm,  # [x1/W, y1/H, x2/W, y2/H]
        "scores": yolo_scores,
        "yolo_output_units": y_units,
        "yolo_output_coords": y_coords,
        "normalized_by": ["width", "height"]
    }
    with open("yolo_out.json", "w") as f:
        json.dump(yolo_payload, f)
    print("Saved YOLO bboxes to yolo_out.json")

    # 3) motionbert_out.json : same structure as the main 3D output
    with open("motionbert_out.json", "w") as f:
        json.dump(result, f)
    print("Saved 3D pose duplicate to motionbert_out.json")

if __name__ == "__main__":
    main()



"""
End-to-end data handling (I/O shapes, formats, and flow)

Below is a concise but exhaustive map of what goes in/out at every stage, how tensors are shaped, which coordinate systems/units are used, and what files get written.

0) Video input & global conventions

Video input: --video (path).

Read via OpenCV cv2.VideoCapture.

Frame size: in_w × in_h (pixels), FPS detected (fallback 30.0).

Coordinate system (pixels): origin at top-left; x→right, y→down.

Normalization rules used in this script:

BBoxes (saved): normalized by the original frame size → [x1/W, y1/H, x2/W, y2/H].

2D keypoints (COCO/H36M): stored in pixels (original frame space), plus a confidence per joint.

MB input 2D coordinates: (x,y) normalized to [-1,1] using:

s = min(W, H)/2, center (cx, cy) = (W/2, H/2)

x’ = (x - cx)/s, y’ = (y - cy)/s

3D output (MotionBERT): model-relative units (as produced by the ONNX), optionally root-relative if --rootrel.

1) YOLOv8 (person detector)
Inputs

Model path: --yolo (or from --cfg_yolo).

Providers: --providers (e.g., CPUExecutionProvider, CUDA/TensorRT if available).

Preprocessing:

Letterbox to (lb_h, lb_w) (defaults 640×640 unless overridden by YOLO cfg: model.input.{height,width} or {target_height,target_width}).

Convert BGR→RGB.

Scale to [0,1], NHWC→NCHW, add batch: input tensor shape [1, 3, lb_h, lb_w], float32.

Outputs (as expected by the code)

The ONNX typically returns one tensor of shape like [1, 84, N] (or [1, N, 84]).
The code transposes to get preds: [N, 84] where:

preds[:, :4] = center-x, center-y, width, height (in letterbox space; possibly normalized depending on model export),

preds[:, 4:] = class scores/logits for C classes (here 80 → 84 = 4 + 80).

Class activation: sigmoid by default (can be changed via cfg).

Person class id: default 0 (can be overridden by cfg).

Postprocess (inside script)

Convert cx,cy,w,h → x1,y1,x2,y2 in letterbox space.

If configured “normalized”, multiply by (lb_w, lb_h) to get letterbox pixels.

Map letterbox coords back to original frame pixels using the stored scale r and pad (dw, dh).

Clip to frame bounds.

NMS with --person_conf (default 0.25) and --person_iou (default 0.45); keep top person box.

Fallbacks: if no detection this frame:

use previous frame’s bbox if available,

else use a heuristic box [0.25W, 0.1H, 0.75W, 0.9H] with score 0.0.

Per-frame data produced

Working (for next stage): a single person bbox in original pixel space: [x1, y1, x2, y2] (float32).

Logged (debug_dir/detections.jsonl): one JSON per frame with:

{
  "t": <int>,
  "bbox": [x1/W, y1/H, x2/W, y2/H],
  "score": <float>,
  "yolo_output_units": "<'normalized' or 'pixels'>",
  "yolo_output_coords": "<'letterbox' or 'original'>"
}


Final aggregate (yolo_out.json):

{
  "video": "<basename>",
  "frames": <int>,
  "bboxes_norm": [[x1/W, y1/H, x2/W, y2/H], ...],
  "scores": [score, ...],
  "yolo_output_units": "...",
  "yolo_output_coords": "...",
  "normalized_by": ["width", "height"]
}

2) RTMPose (2D keypoints on a cropped person)
Crop & resize

crop_to_aspect(frame, det_box, out_hw=(256,192), scale=1.25):

Expands the bbox by scale, adjusts to aspect 256:192, clips to image.

Returns:

crop_bgr: [256, 192, 3] BGR uint8

rect = (rx, ry, rw, rh) = crop region in original frame pixels

Inputs

Model path: --rtmpose (or from --cfg_rtm).

Providers: --providers.

Preprocessing (selectable): --rtm_preproc in {rgb_ms, bgr_ms, rgb_255, bgr_255}

*_ms → mean/std ImageNet style by default: mean [123.675,116.28,103.53], std [58.395,57.12,57.375]

*_255 → scaled to [0,1]

Channel order according to the mode name (rgb_* vs bgr_*)

Input tensor: [1, 3, 256, 192], float32.

Outputs (SimCC heads)

Model returns two tensors: simcc_x, simcc_y of shapes [1, K, Lx], [1, K, Ly]:

K = 17 (COCO-17 joints).

Lx, Ly are the 1D classification lengths along x and y.
The code softmaxes on axis 2 and argmax to get positions.

Decoding to 2D coords

split_ratio = --simcc_ratio (default 2.0).

x̂ = argmax(softmax(simcc_x)) / split_ratio

ŷ = argmax(softmax(simcc_y)) / split_ratio

This yields crop-local coordinates in crop pixel units where the range aligns with the crop size (192 for x, 256 for y) given the ratio.

Confidence per joint:

conf = sqrt(max_prob_x * max_prob_y), per joint, in [0,1].

Mapping crop→original frame

With rect = (rx, ry, rw, rh):

x_img = rx + x̂ * (rw / 192.0)

y_img = ry + ŷ * (rh / 256.0)

COCO-17 keypoints are then stacked:

coco_xy: [17, 2] (pixels, original image)

coco_conf: [17]

Combined per frame as [17, 3] = (x, y, conf).

COCO→H36M remapping

Script converts COCO-17 → H36M-17 with a deterministic mapping and derived joints:

Joints include Pelvis (mid-hip), Spine1, Neck, Head (eyes mid or nose), Site (Nose), shoulders/elbows/wrists, hips/knees/ankles.

Produces per frame:

h36m_xy: [17, 2] (pixels)

h36m_conf: [17]

Combined [17, 3].

Per-sequence arrays (saved as NumPy)

debug/coco_2d.npy: shape [F, 17, 3], float32 (x,y in pixels, conf).

debug/h36m_2d.npy: shape [F, 17, 3], float32.

JSON export (added)

rtm_out.json:

{
  "video": "<basename>",
  "frames": F,
  "coco_order": ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"],
  "coords_2d": [[[x,y,conf],... 17], ... F]
}

3) MotionBERT (lifting 2D→3D over a sequence)
Input preparation

Temporal length: T = 243 (fixed).

Start from H36M-17 per-frame array seq_h36m_xyc: [F, 17, 3].

Sequence length handling:

If --wrap_pad:

If F < T: pad by repeating the last frame until length T.

If F > T: truncate to the first T.

Else: enforce F == T (error otherwise).

Normalization to [-1,1] (per entire video frame size):

Apply the global normalization (see §0) to the x,y channels (pixels → normalized).

Keep the confidence as the 3rd channel.

Final MotionBERT input:

mb_in_seq: shape [T, 17, 3], float32

channel 0..1: normalized x,y in [-1,1]

channel 2: confidence in [0,1]

ONNX input: [1, T, 17, 3] (mb_in_seq[None, ...]), float32.

Outputs

ONNX returns [1, T, 17, 3], float32.

Script takes [0] → X3D: [T, 17, 3].

If --rootrel: set joint 0 (Pelvis) to (0,0,0) for every frame.

Units: model-intrinsic (often mm or normalized; depends on the trained export). The script does not re-scale to camera/world units.

Files saved

debug/mb_input_seq.npy: [T, 17, 3] (normalized 2D + conf).

debug/mb_output_3d.npy: [T, 17, 3] (3D).

Primary JSON (existing, path via --out, default out_3d.json) and duplicate JSON (motionbert_out.json) share the same schema:

{
  "video": "<basename>",
  "T": 243,
  "h36m_order": ["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"],
  "coords_3d": [[[x,y,z], ... 17], ... T]
}

4) Overlay (optional) & debugging artifacts

Overlay video (--save_overlay):

Starts at frame --overlay_start.

Draws H36M-17 skeleton (green lines; red joints) and current YOLO bbox (blue rectangle).

Codec: mp4v, saved as debug/overlay.mp4, frame size in_w × in_h, FPS from input.

Debug files in --debug_dir (when --debug):

io_introspect.txt → environment info & video path.

model_io.json → ONNX input/output names/shapes (as reported by ORT).

config_resolved.json → final CLI/config values.

config_checklist.json → sanity view across YOLO/RTM/MB config aspects.

yolo_decode.txt → per-frame candidate/score trace (if enabled).

detections.jsonl → per-frame normalized bbox + score (see §1).

Optional crop_XXXXX.jpg → first --dump_crops crops.

Quick stats (computed but not echoed):

frames, coco_conf_mean, coco_x_ptp, coco_y_ptp, seen_person.

5) Joint orders (for reference)

COCO-17 (used by RTMPose outputs):
["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"]

H36M-17 (used for MotionBERT I/O & overlays):
["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]

6) Edge cases & fallbacks

No person detected in a frame:

Reuse previous bbox, else use heuristic bbox and continue the pipeline (keeps sequences continuous).

Missing/short sequences for MotionBERT:

If --wrap_pad, pad to length 243 using last valid frame (or zeros if there’s no data).

If not --wrap_pad and F != 243, the script exits with error.

7) Data types summary

Model inputs/outputs: float32.

All NumPy artifacts: float32.

JSON: standard Python numeric types (ints/floats), list-of-lists for arrays.
"""