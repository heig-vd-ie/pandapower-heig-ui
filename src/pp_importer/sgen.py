import dataframely as dy
import polars as pl


class Sgen(dy.Schema):
    """
    Model representing a static generator (sgen) in pandapower.
    """

    name = dy.String(
        nullable=True,
        description="Name of the static generator"
    )
    bus = dy.UInt32(
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )
    p_mw = dy.Float64(
        nullable=False,
        description="Active power in MW",
        metadata={"default": 0.0}
    )
    q_mvar = dy.Float64(
        nullable=True,
        description="Reactive power in MVAr",
        metadata={"default": 0.0}
    )
    sn_mva = dy.Float64(
        nullable=True,
        min=0,
        description="Nominal power in MVA"
    )
    scaling = dy.Float64(
        nullable=False,
        min=0,
        description="Scaling factor for the static generator",
        metadata={"default": 1.0}
    )
    type = dy.Enum(
        categories=["PV", "WP", "CHP", "SGEN"],
        nullable=True,
        description="Type of the static generator"
    )
    in_service = dy.Bool(
        nullable=False,
        description="Indicates if the static generator is in service",
        metadata={"default": True}
    )
    max_p_mw = dy.Float64(
        nullable=True,
        description="Maximum active power in MW"
    )
    min_p_mw = dy.Float64(
        nullable=True,
        description="Minimum active power in MW"
    )
    max_q_mvar = dy.Float64(
        nullable=True,
        description="Maximum reactive power in MVAr"
    )
    min_q_mvar = dy.Float64(
        nullable=True,
        description="Minimum reactive power in MVAr"
    )
    current_source = dy.Bool(
        nullable=False,
        description="Indicates if the static generator is controllable",
        metadata={"default": False}
    )
    controllable = dy.Bool(
        nullable=False,
        description="Indicates if the static generator is controllable",
        metadata={"default": False}
    )
    profile_mapping = dy.String(
            nullable=True,
            description="Indicates if the profile mapping is used for the static generator",
        )
    
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )
