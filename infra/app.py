#!/usr/bin/env python3
"""CDK app entry point for the energy analysis deployment."""
import os
import aws_cdk as cdk
from stack import EnergyAnalysisStack

app = cdk.App()
EnergyAnalysisStack(app, "EnergyAnalysis",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT", "627266360979"),
        region=os.environ.get("CDK_DEFAULT_REGION", "eu-west-2"),
    ),
)
app.synth()
