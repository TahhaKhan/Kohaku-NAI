from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    neg_prompt: str
    seed: int
    scale: float
    width: int
    height: int
    steps: int
    sampler: str
    schedule: str
    smea: bool = False
    dyn: bool = False
    dyn_threshold: bool = False
    cfg_rescale: float = 0.0
    img_sub_folder: str = ""
    extra_infos: str = ""
    priority: int = 1
    model: str = "nai-diffusion-3"
    variety: bool = False
    reference_image_multiple: list[str] = []
    reference_information_extracted_multiple: list[float] = []
    reference_strength_multiple: list[float] = []
    characterPrompts: list[dict] = Field(default_factory=list)
