from dataclasses import dataclass
from pathlib import Path

import zarr
from utils import convert_to_tmp


@dataclass
class TestFiles:
    sensor: str
    mode: str
    ref_input_path: str
    geozarr_path: str

    def __post_init__(self) -> None:
        """Runs automatically right after __init__ - the idiomatic place to
        validate a dataclass's fields as soon as it's constructed."""
        self.validate_model()

    def validate_model(self) -> None:
        # validate geozarr path
        geozarr = Path(self.geozarr_path)
        if not geozarr.exists():
            raise FileNotFoundError(f"geozarr_path does not exist: {geozarr}")
        try:
            zarr.open_group(geozarr, mode="r")
        except Exception as e:
            raise ValueError(f"geozarr_path is not a readable zarr store: {geozarr}") from e

        # validate and checkif zarr or safe
        inputpath = Path(self.ref_input_path)
        if not inputpath.exists():
            raise FileNotFoundError(f"inputpath does not exist: {inputpath}")

        if inputpath.suffix == ".zarr":
            try:
                zarr.open_group(inputpath, mode="r")
                self.input_path = self.ref_input_path

            except Exception as e:
                raise ValueError(f"inputpath is not a readable zarr store: {inputpath}") from e

        elif inputpath.suffix in [".SAFE", ".SEN3"]:
            # will need to convert the file to geozarr in /tmp/ and compare it then..
            import os

            tmpdir = Path(f"/tmp/{os.urandom(8).hex()}")
            tmpdir.mkdir(exist_ok=True)
            outpath = str(tmpdir / f"{inputpath.stem}.zarr")

            print(f"converting to: {outpath}")

            convert_to_tmp(input_path=self.ref_input_path, output_path=outpath)

            try:
                zarr.open_group(outpath, mode="r")

                # overwrite the orginal input path
                self.input_path = outpath

            except Exception as e:
                raise ValueError(f"inputpath is not a readable zarr store: {inputpath}") from e

        return


S2L2AFILES = TestFiles(
    sensor="S2",
    mode="L2A",
    ref_input_path="/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.zarr",
    # ref_input_path="/home/samuel/data/samples/cpm_v300rc4a/safe_products/S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.SAFE",
    geozarr_path="/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.zarr",
)

S2L1CFILES = TestFiles(
    sensor="S2",
    mode="L1C",
    ref_input_path="/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.zarr",
    # ref_input_path="/home/samuel/data/samples/cpm_v300rc4a/safe_products/S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.SAFE",
    geozarr_path="/home/samuel/data/samples/cpm_v300rc4a/converted_zarr_stores/refactored_S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.zarr",
)
