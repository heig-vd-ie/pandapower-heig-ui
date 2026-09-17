import dataframely as dy


class ExtGrid(dy.Schema):
    """
    Dataframely model for pandapower external grid object.
    """

    name = dy.String(
        nullable=True,
        description="Name of the external grid"
    )
    bus = dy.UInt32(
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )
    vm_pu = dy.Float64(
        nullable=True,
        min=0,
        description="Voltage magnitude in per unit",
        metadata={"default": 1.0}
    )
    va_degree = dy.Float64(
        nullable=True,
        description="Voltage angle in degrees",
        metadata={"default": 0.0}
    )
    in_service = dy.Bool(
        nullable=False,
        description="Indicates if the equipment is in service",
        metadata={"default": True}
    )
    slack_weight = dy.Float64(
        nullable=False,
        description="Slack weight",
        metadata={"default": 1.0}
    )

    s_sc_max_mva = dy.Float64(
        nullable=True,
        description="Maximum short-circuit power in MVA"
    )
    s_sc_min_mva = dy.Float64(
        nullable=True,
        description="Minimum short-circuit power in MVA"
    )
    # max_p_mw = dy.Float64(
    #     nullable=True,
    #     min=0,
    #     description="Maximum active power in MW"
    # )
    # min_p_mw = dy.Float64(
    #     nullable=True,
    #     min=0,
    #     description="Minimum active power in MW"
    # )
    # max_q_mvar = dy.Float64(
    #     nullable=True,
    #     description="Maximum reactive power in MVar"
    # )
    # min_q_mvar = dy.Float64(
    #     nullable=True,
    #     description="Minimum reactive power in MVar"
    # )
    rx_max = dy.Float64(
        nullable=True,
        description="Maximum R/X ratio"
    )
    rx_min = dy.Float64(
        nullable=True,
        description="Minimum R/X ratio"
    )
    r0x0_max = dy.Float64(
        nullable=True,
        description="Maximum R0/X0 ratio"
    )
    x0x_max = dy.Float64(
        nullable=True,
        description="Maximum X0/X ratio"
    )
    # controllable = dy.Bool(
    #     nullable=True,
    #     description="Indicates if the external grid is controllable"
    # )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )

