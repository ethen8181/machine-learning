"""Utilities for converting existing posts to Airbnb Knowledge Repo format."""

import codecs
import io
import os
import re
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
    TEMPLATE_PATH = 'header_template.yaml'

    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError('Path does not exist: {path}'.format(path=path))
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
        """Grab the date of creation through git log."""
        cmd = 'git log --diff-filter=A --follow --format=%aD -1 -- {path}'.format(path=self.path)
        return self._git_date_cmd(cmd)

    @property
    def date_updated(self):
        """Grab the last date modified through git log."""
        cmd = 'git log -1 --format=%cd {path}'.format(path=self.path)
        return self._git_date_cmd(cmd)

    @property
    def filename(self):
        """Return name of the file."""
        _, filename = os.path.split(self.path)
        return filename

    @property
    def title(self):
        """Create title for notebook."""
        # TODO: Make a smarter title. We can probably parse the title from the notebook directly.
        title, _ = os.path.splitext(self.filename)
        return title

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

    @property
    def tags(self):
        """Create tags for notebook."""
        # for now, let's just use the directory names as tags
        # for the future, we can fill these in manually or use fuzzy matching on a list of common data science topics
        subdirs = self.subdir.split('/')
        tags = map(lambda x: '- ' + x, subdirs)
        return '\n'.join(tags)

    def construct_header(self):
        with open(self.TEMPLATE_PATH, 'r') as f:
            header = f.read()

        header = header.format(
            title=self.title,
            authors='Author',
            tags=self.tags,
            created_at=self.date_created,
            updated_at=self.date_updated
        )

        return header

    @abstractmethod
    def convert(self):
        raise NotImplementedError


class RmdConverter(KnowledgeRepoConverter):

    ENCODING = 'utf-8'

    def __init__(self, path, new_dir=None):
        super().__init__(path=path)
        self.new_dir = new_dir
        with io.open(self.path, 'r', encoding=self.ENCODING) as file:
            self.notebook = file.read()

    @property
    def title(self):
        regex = r"title: '.*'"  # TODO: match both single and double quotes
        matches = re.search(regex, self.notebook)
        if matches is None:
            title = super().title
        else:
            title = matches.group(0).replace('title: ', '').replace("'", '')
        return title

    def convert(self, inplace=False):
        """Convert .Rmd to Airbnb Knowledge Repo format.

        Attributes:
            inplace: bool
                If True, modifies the file in place. Otherwise, writes file to self.new_dir
        """
        header = self.construct_header()
        github_link = 'See the original file at: {link}'.format(link=self.github_link)

        converted_notebook = '\n'.join([header, github_link, self.notebook])
        converted_notebook.encode('utf-8')

        with codecs.open('BLAHBLAH.txt', 'w', self.ENCODING) as file:
            file.write(converted_notebook)


class IpynbConverter(KnowledgeRepoConverter):
    pass


def convert_all_posts(self, dir_path):
    pass


if __name__ == "__main__":
    PATH = '/Users/eric/Documents/machine-learning/linear_regression/linear_regession.Rmd'
    converter = RmdConverter(PATH)

    print('Date created:', converter.date_created)
    print('Date updated:', converter.date_updated)
    print('Filename:', converter.filename)
    print('Title:', converter.title)
    print('Subdir:', converter.subdir)
    print('Github link:', converter.github_link)

    print('Tags:')
    print(converter.tags)

    print('Header:')
    print(converter.construct_header())

    converter.convert()
