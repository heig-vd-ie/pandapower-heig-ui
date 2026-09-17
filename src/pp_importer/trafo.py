import dataframely as dy
import polars as pl
class Trafo(dy.Schema):
    """
    Pandapower Transformer Model
    """

    name = dy.String(
        nullable=True,
        description="Name of the transformer"
    )
    std_type = dy.String(
        nullable=True,
        description="Transformer standard type name",
    )
    hv_bus = dy.UInt32(
        nullable=False,
        min=0,
        description="High voltage bus ID"
    )
    lv_bus = dy.UInt32(
        nullable=False,
        min=0,
        description="Low voltage bus ID"
    )
    sn_mva = dy.Float64(
        nullable=False,
        min=0,
        description="Rated power of the transformer in MVA"
    )
    vn_hv_kv = dy.Float64(
        nullable=False,
        min=0,
        description="Rated voltage on high voltage side in kV"
    )
    vn_lv_kv = dy.Float64(
        nullable=False,
        min=0,
        description="Rated voltage on low voltage side in kV"
    )

    vk_percent = dy.Float64(
        nullable=False,
        min=0,
        description="Short-circuit voltage in percent"
    )
    vkr_percent = dy.Float64(
        nullable=False,
        min=0,
        description="Short-circuit losses in percent"
    )
    pfe_kw = dy.Float64(
        nullable=False,
        min=0,
        description="Iron losses in kW"
    )
    i0_percent = dy.Float64(
        nullable=False,
        min=0,
        description="Open-circuit current in percent"
    )

    vk0_percent = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence relative sort-circuit voltage in percent"
    )
    vkr0_percent = dy.Float64(
        nullable=True,
        min=0,
        description="Real part of zero sequence relative short-circuit voltage"
    )
    mag0_percent = dy.Float64(
        nullable=True,
        min=0,
        description="Short-circuit voltage in percent"
    )
    mag0_rx = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence magnetizing impedance/ vk0"
    )
    si0_hv_partial = dy.Float64(
        nullable=True,
        description="Distribution of zero sequence leakage impedances for HV side"
    )

    df = dy.Float64(
        nullable=False,
        min=0,
        max=1.0,
        description="Derating factor",
        metadata={"default": 1.0}
    )
    parallel = dy.UInt32(
        nullable=False,
        min=1,
        description="Number of parallel Trafo systems",
        metadata={"default": 1}
    )
    in_service = dy.Bool(
        nullable=False,
        description="Indicates if the equipment is in service",
        metadata={"default": True}
    )

    vector_group = dy.Enum(
        nullable=True,
        categories=["Dyn", "Yyn", "Yzn", "YNyn"],
        description="Vector Groups ( required for zero sequence model of transformer )"
    )
    shift_degree = dy.Float64(
        nullable=False,
        min=0,
        description="Phase shift angle in degrees",
        metadata={"default": 0.0}
    )

    oltc = dy.Bool(
        nullable=False,
        description="Indicates if the equipment is in service",
        metadata={"default": False}
    )
    power_station_unit = dy.Bool(
        nullable=False,
        description="Indicates if the equipment is in service",
        metadata={"default": False}
    )
    tap_changer_type  = dy.Enum(
            nullable=True,
            categories=["Ratio", "Symmetrical", "Ideal", "Tabular"],
            description="specifies the tap changer type. Ratio: ratio tap changer, Symmetrical: symmetrical tap changer, Ideal: ideal tap changer, Tabular: tabular tap changer",
            #  metadata={"default": "Ratio"}
        )
    tap_side = dy.Enum(
        nullable=True,
        categories=["hv", "lv"],
        description="Tap changer side (hv/lv)"
    )
    tap_pos = dy.Int64(
        nullable=True,
        description="Actual tap position"
    )
    tap_neutral = dy.Int64(
        nullable=True,
        description="Neutral tap position"
    )
    tap_min = dy.Int64(
        nullable=True,
        description="Minimum tap position"
    )
    tap_max = dy.Int64(
        nullable=True,
        description="Maximum tap position"
    )
    tap_step_percent = dy.Float64(
        nullable=True,
        min=0,
        description="Tap step size in percent"
    )
    tap_step_degree = dy.Float64(
        nullable=True,
        min=0,
        description="Tap step size in degrees",
    )
    tap_phase_shifter = dy.Bool(
        nullable=True,
        description="Indicates if the transformer is a phase shifter",
    )

    geo = dy.String(
        nullable=True,
        description="Geographical location coordinates for plotting"
    )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )

    @dy.rule()
    def tap_changer_group(cls) -> pl.Expr:
        tap_cols = ["tap_changer_type", "tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_step_degree", "tap_pos", "tap_phase_shifter"]
        return pl.all_horizontal(pl.col(tap_cols).is_null()) | pl.all_horizontal(pl.col(tap_cols).is_not_null())
