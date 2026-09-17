import dataframely as dy
import polars as pl



class Line(dy.Schema):
    name = dy.String(
        nullable=True,
        description="A custom name for this line"
    )
    std_type = dy.String(
        nullable=True,
        description="Transformer standard type name",
    )
    from_bus = dy.UInt32(
        nullable=False,
        min=0,
        description="ID of the bus on one side which the line will be connected with"
    )
    to_bus = dy.UInt32(
        nullable=False,
        min=0,
        description="ID of the bus on the other side which the line will be connected with"
    )
    length_km = dy.Float64(
        nullable=False,
        min=0,
        description="The line length in km"
    )

    r_ohm_per_km = dy.Float64(
        nullable=False,
        min=0,
        description="Line resistance in ohm per km"
    )
    x_ohm_per_km = dy.Float64(
        nullable=False,
        min=0,
        description="Line reactance in ohm per km"
    )
    c_nf_per_km = dy.Float64(
        nullable=False,
        min=0,
        description="Line capacitance (line-to-earth) in nano Farad per km"
    )
    g_us_per_km = dy.Float64(
        nullable=False,
        min=0,
        description="Dielectric conductance in micro Siemens per km",
        metadata={"default": 0}
    )
    r0_ohm_per_km = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence line resistance in ohm per km"
    )
    x0_ohm_per_km = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence line reactance in ohm per km"
    )
    c0_nf_per_km = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence line capacitance in nano Farad per km"
    )
    g0_us_per_km = dy.Float64(
        nullable=True,
        min=0,
        description="Zero sequence dielectric conductance in micro Siemens per km"
    )
    max_i_ka = dy.Float64(
        nullable=False,
        min=0,
        metadata={"default": 1e6},
        description="Maximum thermal current in kilo Ampere"
    )
    max_loading_percent = dy.Float64(
        nullable=False,
        min=0,
        description="Maximum current loading (only needed for OPF)",
        metadata={"default": 100.0}
    )
    section = dy.Float64(
        nullable=True,
        min=0,
        description="Section of the line"
    )
    df = dy.Float64(
        nullable=False,
        min=0,
        max=1.0,
        description="Derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)",
        metadata={"default": 1.0}
    )
    parallel = dy.UInt32(
        nullable=False,
        min=1,
        description="Number of parallel line systems",
        metadata={"default": 1}
    )
    in_service = dy.Bool(
        nullable=False,
        description="True for in_service or False for out of service",
        metadata={"default": True}
    )
    type = dy.Enum(
        nullable=False,
        categories=["ol", "cs"],
        description="Type of line ('ol' for overhead line or 'cs' for cable system)",
        metadata={"default": "ol"}
    )
    geo = dy.String(
        nullable=True,
        description="Geographical location coordinates for plotting"
    )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected"
    )
