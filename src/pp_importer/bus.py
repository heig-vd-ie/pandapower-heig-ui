import dataframely as dy
import polars as pl

class Bus(dy.Schema):
    name = dy.String(
        nullable=True,
        description="The name for this bus"
    )
    vn_kv = dy.Float64(
        nullable=False,
        min=0,
        description="The grid voltage level"
    )
    type = dy.Enum(
        categories=["b", "n", "m"],
        nullable=False,
        description="Type of the bus. 'n' - node, 'b' - busbar, 'm' - muff",
        metadata={"default": "n"}
    )
    zone = dy.String(
        nullable=True,
        description="Grid region"
    )
    in_service = dy.Bool(
        nullable=False,
        description="True for in_service or False for out of service",
        metadata={"default": True}
    )
    max_vm_pu = dy.Float64(
        nullable=True,
        min=0,
        description="Maximum bus voltage in p.u. - necessary for OPF",
        metadata={"default": 1.1}
    )
    min_vm_pu = dy.Float64(
        nullable=True,
        min=0,
        description="Minimum bus voltage in p.u. - necessary for OPF",
        metadata={"default": 0.9}
    )
    
    cn_fk = dy.String(
        nullable=True,
        description="Unique uuid from DataSchema"
    )
    geo = dy.String(
        nullable=True,
        description="Geographical location coordinates for plotting"
    )
    dso_code = dy.String(
        nullable=True,
        description="DSO code of the line"
    )
    id = dy.UInt32(
        primary_key=True,
        nullable=False,
        min=0,
        description="Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected"
    )

