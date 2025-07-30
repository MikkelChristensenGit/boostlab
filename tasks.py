from invoke import task


@task
def precommit(c):
    """
    Run all pre-commit hooks on all files
    Run "invoke precommit"
    """
    c.run("pre-commit run --all-files")
