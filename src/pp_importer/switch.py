import dataframely as dy
import polars as pl



class Switch(dy.Schema):
    """
    Pandapower Switch Object
    """

    name = dy.String(
        nullable=True,
        description="The name of the switch",
    )
    bus = dy.UInt32(
        nullable=False,
        description="The bus to which the switch is connected",
    )
    element = dy.UInt32(
        nullable=False,
        description="The element to which the switch is connected",
    )
    et = dy.Enum(
        nullable=False,
        categories=["b", "l", "t", "t3"],
        description="The type of element (bus or line)",
        metadata={"default": "b"}
    )
    closed = dy.Bool(
        nullable=False,
        description="The status of the switch (open or closed)",
        metadata={"default": True}
    )
    type = dy.Enum(
        nullable=True,
        categories=["CB", "LS", "LBS", "DS"],
        description="The type of switch",
        metadata={"default": "CB"}
    )
    in_ka = dy.Float64(
        nullable=True,
        description="The type of switch",
    )
    z_ohm = dy.Float64(
        nullable=False,
        min=0,
        description="Conducting impedance of the switch",
        metadata={"default": 0}
    )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Unique id used as index by pandapower"
    )
