from invoke import task


@task
def precommit(c):
    """Run all pre-commit hooks on all files"""
    c.run("pre-commit run --all-files")
