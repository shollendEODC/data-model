import xarray as xr


def convert_s1grdh_optimized(dt_input: xr.DataTree,
                                *,
                                enable_sharding: bool,
                                output_path: str,
                                spatial_chunk: int,
                                compression_level: int,
                                validate_output: bool,
                                keep_scale_offset: bool,
                                max_retries: int = 3,
                            ) -> xr.DataTree:
    pass
    return xr.DataTree()