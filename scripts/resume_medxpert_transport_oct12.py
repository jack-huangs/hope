"""等待首遍结束，再最多两遍curl恢复失败请求；成功后离线验收。"""
import argparse
import ctypes
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import prepare_relevance_corpus as base
import prepare_medxpert_oct12 as preparation
import finish_medxpert_oct12 as finish


def curl_request(url, filename, timeout):
    # 与urllib使用相同URL。curl自身无重试，重试次数仍由已有清单逻辑控制。
    result = subprocess.run(["curl.exe", "-L", "--retry", "0", "--max-time", str(timeout),
        "--silent", "--show-error", "--output", str(filename), "--write-out", "%{http_code}", url],
        capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != "200":
        raise RuntimeError(f"curl exit={result.returncode}, HTTP={result.stdout.strip()}; {result.stderr[:150]}")
    raw = filename.read_bytes()
    # 某些中间代理可能给截断正文也返回200；解析失败必须记作失败attempt。
    if "efetch.fcgi" in url:
        root = ET.fromstring(raw)
        if root.tag != "PubmedArticleSet":
            raise ValueError("EFetch did not return a PubmedArticleSet")
    else:
        if "esearchresult" not in json.loads(raw):
            raise ValueError("ESearch response lacks esearchresult")
    return raw


def complete():
    path = preparation.ROOT / "data/medxpert_broad_oct12/preparation_manifest.json"
    return path.exists() and json.loads(path.read_bytes())["status"] == "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, args.wait_pid)
    if handle:
        while kernel.WaitForSingleObject(handle, 5000) == 258:
            pass
        kernel.CloseHandle(handle)
    print("Initial urllib process exited; bounded curl recovery starts", flush=True)
    original = base.request_with_retries
    def annotated(url, prefix, suffix, timeout, retries, backoff, entry, checkpoint):
        before = len(entry["attempts"])
        def save():
            for attempt in entry["attempts"][before:]:
                attempt["transport"] = "curl.exe"
            checkpoint()
        return original(url, prefix, suffix, timeout, retries, backoff, entry, save)
    base.request_once = curl_request
    base.request_with_retries = annotated
    attempts = []
    for number in range(1, 3):
        if complete():
            break
        print(f"curl recovery pass {number}", flush=True)
        record = {"pass": number, "transport": "curl.exe", "started_at": base.now()}
        try:
            preparation.main()
            record["status"] = "returned"
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:300]})
            print(record, flush=True)
        attempts.append(record)
        base.save_json(preparation.ROOT / "runs/medxpert_preparation_oct12/transport_recovery.json", attempts)
    if not complete():
        raise RuntimeError("Two bounded recovery passes exhausted; snapshot remains incomplete")
    finish.main()


if __name__ == "__main__":
    main()
