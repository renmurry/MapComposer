"""Batch: one map per sub-folder, template chosen per folder (or fixed)."""
import os
import re
from dataclasses import replace

from .detect import scan, context
from .templates import suggest
from .compose import compose


def safe_name(s):
    s = re.sub(r"[^\w\- ]+", " ", s)
    s = re.sub(r"[\s_]+", "_", s).strip("_-")
    return s or "map"


def sub_folders(parent):
    return [os.path.join(parent, n) for n in sorted(os.listdir(parent)) if os.path.isdir(os.path.join(parent, n))]


def plan(parent, templates, fixed_id=None):
    """[(folder, scan_result, template_or_None, reason)]"""
    out = []
    for f in sub_folders(parent):
        try:
            s = scan(f)
        except Exception as e:
            out.append((f, None, None, f"cannot read: {e}")); continue
        if not s["rasters"] and not s["vectors"]:
            out.append((f, s, None, "no map data")); continue
        if fixed_id:
            t = next((t for t in templates if t["id"] == fixed_id), None)
        else:
            ranked = suggest(templates, context(s))
            t = ranked[0][1] if ranked and ranked[0][0] > 0 else None
        out.append((f, s, t, "ready" if t else "no template fits"))
    return out


def run_batch(parent, templates, job_template, fixed_id=None, log=print, stop_flag=lambda: False):
    """job_template: a Job with shared texts/options; title/subtitle None = use each template's defaults."""
    results = []
    for folder, s, t, reason in plan(parent, templates, fixed_id):
        if stop_flag():
            log("Stopped."); break
        log(f"--- {os.path.basename(folder)}")
        if t is None:
            log(f"   skipped: {reason}"); results.append((folder, "skipped", reason)); continue
        log(f"   template: {t['name']}")
        for r in s["rasters"]:
            if r["role"] != "Ignore":
                log(f"   raster {r['layer'].name}: {r['role']}")
        for v in s["vectors"]:
            if v["role"] != "Ignore":
                log(f"   {v['layer'].name}: {v['role']}" + (f" (by '{v['style_field']}')" if v["style_field"] else ""))
        texts = dict(job_template.texts)
        texts["name"] = s["name"]
        texts["title"] = texts.get("title") or t["text"]["title"]
        if texts.get("subtitle") is None:
            texts["subtitle"] = t["text"]["subtitle"]
        paper = job_template.paper or t["page"]["paper"]
        job = replace(job_template, template=t, rasters=s["rasters"], vectors=s["vectors"], texts=texts,
                      out_base=os.path.join(folder, f"{safe_name(s['name'] + '_' + t['id'])}_{paper}"))
        try:
            r = compose(job, log=lambda m: log("   " + m))
            for w in r["warnings"]:
                log("   note: " + w)
            log("   saved: " + ", ".join(os.path.basename(o) for o in r["outputs"]))
            results.append((folder, "ok", r))
        except Exception as e:
            log(f"   FAILED: {e}"); results.append((folder, "error", str(e)))
    return results
