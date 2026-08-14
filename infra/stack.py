"""
AWS CDK stack for the energy analysis pipeline.

Resources:
  - ECS Fargate task (daily model run)
  - EFS file system (persistent SQLite DB)
  - S3 bucket (static dashboard)
  - CloudFront distribution (CDN)
  - EventBridge rule (10:00 UTC daily schedule)
  - Secrets Manager (API keys)
"""
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_efs as efs,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_ecr_assets as ecr_assets,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class EnergyAnalysisStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── VPC ───────────────────────────────────────────────────────────────
        # Use the account's default VPC — no need to create a dedicated one
        vpc = ec2.Vpc.from_lookup(self, "Vpc", is_default=True)

        # ── EFS (persistent storage for energy.db) ───────────────────────────
        file_system = efs.FileSystem(self, "Efs",
            vpc=vpc,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_30_DAYS,
        )
        access_point = file_system.add_access_point("DataAP",
            path="/data",
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="755"),
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
        )

        # ── S3 bucket (dashboard HTML + charts) ──────────────────────────────
        dashboard_bucket = s3.Bucket(self, "Dashboard",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ── CloudFront ────────────────────────────────────────────────────────
        distribution = cloudfront.Distribution(self, "Cdn",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(dashboard_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            default_root_object="index.html",
        )

        # ── Secrets Manager (API keys) ───────────────────────────────────────
        api_keys = secretsmanager.Secret(self, "ApiKeys",
            description="API keys for energy analysis data sources",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"GIE_API_KEY":"","EIA_API_KEY":"","ENTSOE_API_KEY":""}',
                generate_string_key="_unused",
            ),
        )

        # ── ECS Cluster ──────────────────────────────────────────────────────
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # ── Fargate Task Definition ──────────────────────────────────────────
        task_def = ecs.FargateTaskDefinition(self, "TaskDef",
            cpu=1024,       # 1 vCPU
            memory_limit_mib=4096,  # 4 GB — LightGBM + Optuna needs headroom
        )

        # Mount EFS
        task_def.add_volume(
            name="data",
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(
                    access_point_id=access_point.access_point_id,
                    iam="ENABLED",
                ),
            ),
        )

        # Grant EFS access to the task role
        file_system.grant_root_access(task_def.task_role)
        file_system.connections.allow_default_port_from(
            ec2.Peer.ipv4(vpc.vpc_cidr_block)
        )

        # Container
        container = task_def.add_container("App",
            image=ecs.ContainerImage.from_asset("..", platform=ecr_assets.Platform.LINUX_AMD64),  # Build from project root Dockerfile
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="energy",
                log_retention=logs.RetentionDays.TWO_WEEKS,
            ),
            environment={
                "DASHBOARD_BUCKET": dashboard_bucket.bucket_name,
                "CLOUDFRONT_DISTRIBUTION_ID": distribution.distribution_id,
                "DB_PATH": "/data/energy.db",
                "CHARTS_DIR": "/data/charts",
                "DASHBOARD_PATH": "/data/index.html",
            },
            secrets={
                "GIE_API_KEY": ecs.Secret.from_secrets_manager(api_keys, "GIE_API_KEY"),
                "EIA_API_KEY": ecs.Secret.from_secrets_manager(api_keys, "EIA_API_KEY"),
                "ENTSOE_API_KEY": ecs.Secret.from_secrets_manager(api_keys, "ENTSOE_API_KEY"),
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(api_keys, "OPENAI_API_KEY"),
            },
            stop_timeout=Duration.seconds(120),
        )
        container.add_mount_points(
            ecs.MountPoint(
                container_path="/data",
                source_volume="data",
                read_only=False,
            ),
        )

        # Grant S3 write + CloudFront invalidation
        dashboard_bucket.grant_read_write(task_def.task_role)
        task_def.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudfront:CreateInvalidation"],
                resources=[f"arn:aws:cloudfront::{self.account}:distribution/{distribution.distribution_id}"],
            ),
        )

        # ── EventBridge Schedule (13:00 UTC daily — after EPEX D+1 auction clears ~12:00 CET)
        rule = events.Rule(self, "DailyRun",
            schedule=events.Schedule.cron(minute="0", hour="13"),
            description="Run energy analysis pipeline daily at 13:00 UTC (after EPEX D+1 clears)",
        )

        rule.add_target(
            targets.EcsTask(
                cluster=cluster,
                task_definition=task_def,
                subnet_selection=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PUBLIC,
                ),
                assign_public_ip=True,
            ),
        )
