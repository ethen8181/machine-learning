"""Utilities for converting existing posts to Airbnb Knowledge Repo format."""

import os
import subprocess
from abc import abstractmethod
from dateutil import parser as date_parser


class KnowledgeRepoConverter:

    """Converts a post to knowledge_repo format.

    1. Adds YAML style header
    2. Extracts metadata on the post: date created, date updated
    3. Embeds link to github link (for access to additional files)
    """
    BASE_URL = 'https://github.com/ethen8181/machine-learning'
    REPO_NAME = 'machine-learning'
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, path):
        self.path = path

    def _git_date_cmd(self, cmd):
        """Run bash command to retrieve and format date string."""
        assert isinstance(cmd, str)
        date_str = subprocess.check_output(cmd.split(' '))
        date_dt = date_parser.parse(date_str)
        formatted_date = date_dt.strftime(self.DATE_FORMAT)
        return formatted_date

    @property
    def date_created(self):
        """Grabs the date of creation through git log"""
        cmd = 'git log --diff-filter=A --follow --format=%aD -1 -- {path}'.format(path=self.path)
        return self._git_date_cmd(cmd)

    @property
    def date_updated(self):
        """Grabs the last date modified through git log"""
        cmd = 'git log -1 --format=%cd {path}'.format(path=self.path)
        return self._git_date_cmd(cmd)

    @property
    def filename(self, ext=True):
        """Returns name of the file."""
        _, filename = os.path.split(self.path)
        if not ext:
            filename, _ = os.path.splitext(filename)
        return filename

    @property
    def subdir(self):
        _, path_within_repo = self.path.split(self.REPO_NAME)
        subdir, _ = os.path.split(path_within_repo)
        if subdir[0] == '/':
            subdir = subdir[1:]
        return subdir

    @property
    def github_link(self):
        link_components = [self.BASE_URL, 'blob', 'master', self.subdir, self.filename]
        return '/'.join(link_components)

    def construct_yaml(self):
        pass

    @abstractmethod
    def convert(self, path):
        pass


class RmdConverter(KnowledgeRepoConverter):
    pass


class IpynbConverter(KnowledgeRepoConverter):
    pass


def convert_all_posts(self, dir_path):
    pass


if __name__ == "__main__":
    PATH = '/Users/eric/Documents/machine-learning/text_classification/naive_bayes/naive_bayes.ipynb'
    converter = KnowledgeRepoConverter(PATH)

    print('Date created:', converter.date_created)
    print('Date updated:', converter.date_updated)
    print('Filename:', converter.filename)
    print('Subdir:', converter.subdir)
    print('Github link:', converter.github_link)
