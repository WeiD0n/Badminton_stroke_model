
"""add password recovery fields

Revision ID: 55965b12fbe9
"""

from alembic import op
import sqlalchemy as sa


revision = "55965b12fbe9"
down_revision = "c3c3b489b51f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("recovery_code_hash", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column("recovery_code_used_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column(
            "recovery_code_failed_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0"
        )
    )
    op.add_column(
        "users",
        sa.Column("recovery_code_locked_until", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column(
            "token_version",
            sa.Integer(),
            nullable=False,
            server_default="0"
        )
    )


def downgrade():
    op.drop_column("users", "token_version")
    op.drop_column("users", "recovery_code_locked_until")
    op.drop_column("users", "recovery_code_failed_attempts")
    op.drop_column("users", "recovery_code_used_at")
    op.drop_column("users", "recovery_code_hash")
