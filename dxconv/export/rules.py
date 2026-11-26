"""export 설정 검증 규칙.

위반 시 명확한 메시지를 담은 RuleError 를 던진다. 일부 항목은 경고만 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dxconv import DxconvError
from dxconv.config import ExportCfg, PostCfg
from dxconv.export.inspect_pt import PtInfo

DEFAULT_OPSET_RANGE = (11, 21)
STRIDE = 32


class RuleError(DxconvError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = "export 설정 규칙 위반:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


@dataclass
class RuleResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_error(self) -> None:
        if self.errors:
            raise RuleError(self.errors)


def check_export_rules(
    export: ExportCfg,
    post: PostCfg | None = None,
    compat: dict[str, Any] | None = None,
    pt_info: PtInfo | None = None,
) -> RuleResult:
    r = RuleResult()
    lo, hi = tuple((compat or {}).get("opset_range") or DEFAULT_OPSET_RANGE)

    if export.half:
        r.errors.append(
            "half=True 는 사용할 수 없습니다: DX-COM 이 INT8 양자화를 수행하므로 FP32 ONNX 가 필요합니다."
        )
    if not (lo <= export.opset <= hi):
        r.errors.append(f"opset={export.opset} 이(가) 지원 범위 [{lo}, {hi}] 밖입니다 (compat.yaml).")
    if export.dynamic:
        r.errors.append("dynamic=True 는 사용할 수 없습니다: NPU 는 정적 입력 shape 만 지원합니다.")
    if export.batch != 1:
        r.errors.append(f"batch={export.batch}: NPU 변환은 batch=1 정적 shape 만 지원합니다.")
    if export.nms and export.end2end:
        r.errors.append(
            "nms=True 와 end2end=True 는 동시에 사용할 수 없습니다 "
            "(end2end one2one head 는 NMS 가 필요 없음)."
        )
    if export.imgsz % STRIDE != 0:
        r.errors.append(f"imgsz={export.imgsz} 는 {STRIDE} 의 배수여야 합니다.")

    # max_det: 경고만
    if post is not None:
        if post.max_det is not None and post.max_det != export.max_det:
            r.warnings.append(
                f"export.max_det({export.max_det}) 와 postprocess.max_det({post.max_det}) 가 다릅니다. "
                "PT/ONNX/DXNN 평가 결과가 일치하지 않을 수 있습니다."
            )
    if export.end2end and export.max_det != 300:
        r.warnings.append(
            f"end2end 출력은 max_det={export.max_det} 개로 고정됩니다. "
            "평가 측(device_agent/PT 평가)에서도 같은 max_det 를 사용하는지 확인하세요."
        )

    if pt_info is not None:
        if export.end2end and not pt_info.end2end_supported:
            r.errors.append(
                f"end2end=True 이지만 모델 head({pt_info.head_type}) 가 end2end(one2one) 를 "
                "지원하지 않습니다. end2end=false, nms 설정을 검토하세요."
            )
        if pt_info.task not in (None, "detect"):
            r.warnings.append(f"task={pt_info.task}: dxconv 는 detect 모델 기준으로 검증되었습니다.")
        tr = pt_info.train_imgsz
        if isinstance(tr, int) and tr != export.imgsz:
            r.warnings.append(f"학습 imgsz({tr}) 와 export imgsz({export.imgsz}) 가 다릅니다.")
    return r
