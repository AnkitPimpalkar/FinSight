from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    raw_data_file: str
    period: str
    interval: str

