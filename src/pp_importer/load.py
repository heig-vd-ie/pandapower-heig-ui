import dataframely as dy
import polars as pl



class Load(dy.Schema):
    """
    Represents a load object in pandapower with default values and constraints.
    """

    name = dy.String(
        nullable=True,
        description="Name of the load"
    )
    bus = dy.UInt32(
        nullable=False,
        min=0,
        description="Bus number to which the load is connected"
    )
    p_mw = dy.Float64(
        nullable=True,
        description="Active power in MW",
        metadata={"default": 0.0}
    )
    q_mvar = dy.Float64(
        nullable=True,
        description="Reactive power in MVAr",
        metadata={"default": 0.0}
    )
    sn_mva = dy.Float64(
        nullable=False,
        min=0.0,
        description="Reactive power in MVAr",
        metadata={"default": 0.0}
    )
    const_z_p_percent = dy.Float64(
        nullable=False,
        min=0.0,
        max=100.0,
        description="percentage of p_mw that is associated to constant impedance load at rated voltage [%]",
        metadata={"default": 0.0}
    )
    const_i_p_percent = dy.Float64(
        nullable=False,
        min=0.0,
        max=100.0,
        description="percentage of p_mw that is associated to constant current load at rated voltage [%]",
        metadata={"default": 0.0}
    )
    const_z_q_percent = dy.Float64(
            nullable=False,
            min=0.0,
            max=100.0,
            description="percentage of q_mvar that is associated to constant impedance load at rated voltage [%]",
            metadata={"default": 0.0}
        )
    const_i_q_percent = dy.Float64(
        nullable=False,
        min=0.0,
        max=100.0,
        description="percentage of q_mvar that is associated to constant current load at rated voltage [%]",
        metadata={"default": 0.0}
    )

    scaling = dy.Float64(
        nullable=False,
        min=0.0,
        description="Scaling factor for the load",
        metadata={"default": 1.0}
    )
    in_service = dy.Bool(
        nullable=False,
        description="Indicates if the load is in service",
        metadata={"default": True}
    )
    type = dy.Enum(
        categories=["wye", "delta"],
        nullable=True,
        description="Type of the load"
    )
    controllable = dy.Bool(
        nullable=False,
        description="Indicates if the load is controllable",
        metadata={"default": False}
    )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )
    profile_mapping = dy.String(
        nullable=True,
        description="Indicates if the profile mapping is used for the static generator",
    )
        
