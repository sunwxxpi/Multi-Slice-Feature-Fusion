"""aggregate_5fold_results.py 의 NETCLASS 매핑이 4개 decoder 경로를 모두 만드는지 검증.

포뮬러를 다시 구현하지 않고 모듈의 main() 을 실제로 실행해, 존재하지 않는 결과 디렉터리에 대해
찍히는 [warn] 메시지의 test_dir 경로 문자열로 NetClass_encoder 서브디렉터리를 확인한다.
"""
import contextlib
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/tmp/claude-1004/-home-psw-SAU-Net/797139ad-2f95-45db-a0f1-61976fc9b863/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)

sys.path.insert(0, REPO)
import aggregate_5fold_results as agg

CASES = [
    ("unet", "resnet50_sa", "Unet_resnet50_sa"),
    ("segformer", "resnet50_sa", "Segformer_resnet50_sa"),
    ("emcad", "pvt_v2_b2", "EMCADNet_pvt_v2_b2"),
    ("emcad_sa", "pvt_v2_b2", "EMCAD_SA_Net_pvt_v2_b2"),
]

for decoder, encoder, expected_sub in CASES:
    argv = ["aggregate_5fold_results.py",
            "--decoder", decoder, "--encoder", encoder,
            "--exp_template", "dummy_fold{fold}",
            "--test_log_root", os.path.join(SCRATCH, "no_test_log"),
            "--model_root", os.path.join(SCRATCH, "no_model"),
            "--out", os.path.join(SCRATCH, f"agg_check_{decoder}.md")]
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            agg.main()
    except SystemExit as e:
        raise AssertionError(
            f"{decoder}: argparse 가 --decoder {decoder} 를 거부함 "
            f"(choices 에 없음, exit={e.code}, stderr={err_buf.getvalue()!r})"
        ) from None
    finally:
        sys.argv = old_argv
    out = out_buf.getvalue()
    assert expected_sub in out, f"{decoder}: 경로에 {expected_sub!r} 없음 (출력 앞부분: {out[:300]!r})"

print("OK: aggregate_5fold_results.py 의 NETCLASS 매핑이 unet/segformer/emcad/emcad_sa "
      "4개 경로를 모두 만든다")
