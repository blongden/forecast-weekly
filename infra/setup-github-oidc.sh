#!/usr/bin/env bash
# One-time setup: create GitHub OIDC provider + IAM role for GitHub Actions.
# Run this once from a terminal with AWS admin credentials.
#
# Usage: bash infra/setup-github-oidc.sh

set -euo pipefail

ACCOUNT_ID="627266360979"
REGION="eu-west-2"
GITHUB_ORG="blongden"
GITHUB_REPO="forecast-weekly"
ROLE_NAME="github-actions-forecast-weekly"
ECR_REPO="cdk-hnb659fds-container-assets-${ACCOUNT_ID}-${REGION}"
TASK_DEF_FAMILY="EnergyAnalysisTaskDefE9704C45"
EB_RULE="EnergyAnalysis-DailyRunDEF7747D-KQZLhxQuSBPa"
TASK_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/EnergyAnalysis-TaskDefTaskRole1EDB4A67-OAhlqkMZVzum"
EXEC_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/EnergyAnalysis-TaskDefExecutionRoleB4775C97-xH1D17NUDIrd"

echo "==> Creating GitHub OIDC provider …"
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  2>/dev/null || echo "    (provider already exists — skipping)"

OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

echo "==> Creating IAM role ${ROLE_NAME} …"
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Federated": "${OIDC_ARN}" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*"
        }
      }
    }
  ]
}
EOF
)

aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document "$TRUST_POLICY" \
  2>/dev/null || echo "    (role already exists — updating trust policy)"

aws iam update-assume-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-document "$TRUST_POLICY"

echo "==> Attaching inline policy …"
INLINE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "ECRPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/${ECR_REPO}"
    },
    {
      "Sid": "ECSTaskDef",
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EventBridge",
      "Effect": "Allow",
      "Action": "events:PutTargets",
      "Resource": "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${EB_RULE}"
    },
    {
      "Sid": "PassRoles",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "${TASK_ROLE}",
        "${EXEC_ROLE}"
      ]
    }
  ]
}
EOF
)

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "forecast-weekly-deploy" \
  --policy-document "$INLINE_POLICY"

echo ""
echo "Done. Role ARN:"
aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text
