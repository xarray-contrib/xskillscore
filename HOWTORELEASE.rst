Release Procedure
-----------------

#. Create a new branch ``release-v0.0.x`` with the version for the release.

 * Update ``CHANGELOG.rst``
 * Make sure all new changes, features are reflected in the documentation.

#. Open a new pull request for this branch targeting `master`

#. After all tests pass and the PR has been approved, merge the PR into ``master``

#. Tag a release and push to github::

    $ git tag -a v0.0.17 -m "Version 0.0.17"
    $ git push origin master --tags

#. Build and publish release on PyPI::

    $ git clean -xfd  # remove any files not checked into git
    $ python setup.py sdist bdist_wheel --universal  # build package
    $ twine upload dist/*  # register and push to pypi

#. Update the stable branch (used by ReadTheDocs)::

    $ git checkout stable
    $ git rebase master
    $ git push -f origin stable
    $ git checkout master

#. Go to https://readthedocs.org and add the new version to "Active Versions"
   under the version tab. Force-build "stable" if it isn't already building.

#. Update ``xskillscore`` conda-forge feedstock

 * Fork `xskillscore-feedstock repository <https://github.com/conda-forge/xskillscore-feedstock>`_
 * Clone this fork and edit recipe::

        $ git clone git@github.com:username/xskillscore-feedstock.git
        $ cd xskillscore-feedstock
        $ cd recipe
        $ # edit meta.yaml

 - Update version
 - Get sha256 from pypi.org for `xskillscore <https://pypi.org/project/xskillscore/#files>`_
 - Fill in the rest of information as described `here <https://github.com/conda-forge/xskillscore-feedstock#updating-xskillscore-feedstock>`_

 * Commit and submit a PR
