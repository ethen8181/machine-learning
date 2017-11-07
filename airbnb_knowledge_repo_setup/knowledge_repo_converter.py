"""Utilities for converting existing posts to Airbnb Knowledge Repo format."""

import codecs
import io
import os
import re
import json
import subprocess
from dateutil import parser as date_parser
from random_words import LoremIpsum, RandomNicknames


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
    def authors(self):
        rn = RandomNicknames()
        return '- {name}'.format(name=rn.random_nick(gender='u'))

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
        tags = list(map(lambda x: '- ' + x, subdirs))
        return '\n'.join(tags)

    @property
    def tldr(self):
        li = LoremIpsum()
        return li.get_sentence()
        # return "Fill in tldr. Here's the github link for now: {link}".format(link=self.github_link)

    def construct_header(self):
        with open(self.TEMPLATE_PATH, 'r') as f:
            header = f.read()

        header = header.format(
            title=self.title,
            authors=self.authors,
            tags=self.tags,
            created_at=self.date_created,
            updated_at=self.date_updated
        )

        return header

    def convert(self):
        raise NotImplementedError

    def write(self, inplace=False):
        raise NotImplementedError


class RmdConverter(KnowledgeRepoConverter):

    ENCODING = 'utf-8'

    def __init__(self, path, new_dir=None):
        super().__init__(path=path)
        self.new_dir = new_dir
        with io.open(self.path, 'r', encoding=self.ENCODING) as fp:
            self.notebook = fp.read()

    @property
    def title(self):
        regex = r"title: '.*'"  # TODO: match both single and double quotes
        matches = re.search(regex, self.notebook)
        if matches is None:
            title = super().title
        else:
            title = matches.group(0).replace('title: ', '').replace("'", '')
        return title

    def fix_setwd_calls(self):
        """Remove setwd() calls and insert cell in .Rmd."""
        pass

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

        with codecs.open('BLAHBLAH.txt', 'w', self.ENCODING) as fp:
            fp.write(converted_notebook)


class IpynbConverter(KnowledgeRepoConverter):

    def __init__(self, path, inplace=False):
        super().__init__(path=path)
        with io.open(path, 'r', encoding='utf-8') as fp:
            self.notebook = json.load(fp)
        self.inplace = inplace

    @property
    def tags(self):
        subdirs = self.subdir.split('/')
        tags = list(map(lambda x: '- ' + x, subdirs))
        return tags

    @property
    def write_path(self):
        if self.inplace:
            path = self.path
        else:
            head, ext = os.path.splitext(self.path)
            head += '-converted'
            path = head + ext
        return path

    def construct_header(self):
        """Create a knowledge repo style header as a dictionary."""
        def flatten_list(l):
            flat = []
            for item in l:
                if isinstance(item, list):
                    flat += item
                else:
                    flat.append(item)
            return flat

        header = {
            'cell_type': 'raw',
            'metadata': {},
            'source': []
        }

        header_text = [
            '---',
            'title: {}'.format(self.title),
            'authors:',
            self.authors,
            'tags:',
            self.tags,
            'created_at: {}'.format(self.date_created),
            'updated_at: {}'.format(self.date_updated),
            'tldr: {}'.format(self.tldr),
            '---'
        ]

        header_text = flatten_list(header_text)
        header_text = list(map(lambda x: x + '\n', header_text))
        header['source'] = header_text
        return header

    def construct_github_link_cell(self):
        pass

    def convert(self):
        """Prepend the dictionary header to self.notebook['cells']."""
        self.notebook['cells'] = [self.construct_header()] + self.notebook['cells']

    def write(self):
        with io.open(self.write_path, 'w', encoding='utf-8') as fp:
            json.dump(self.notebook, fp)


def convert_all_posts(path):
    """Recursively walk the root directory, converting and adding .ipynb to the knowledge repo."""
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)]
        for f in files:
            convert_all_posts(f)
    elif '-converted' not in path:
        head, ext = os.path.splitext(path)
        if ext == ".ipynb":
            try:
                converter = IpynbConverter(path, inplace=False)
                converter.convert()
                converter.write()
                add_to_knowledge_repo(converter.write_path)
            except Exception as e:
                print('Skipping: {}'.format(path))
                print(e)


def add_to_knowledge_repo(path):
    REPO = '/Users/eric/Documents/machine-learning/knowledge'

    # determine name of post
    head, _ = os.path.splitext(path)
    _, name = os.path.split(head)
    name = name.replace('-converted', '')
    destination = '{repo}/project/{name}'.format(repo=REPO, name=name)

    # create a run knowledge repo command
    cmd = 'knowledge_repo --repo {repo} add {file_path} -p {destination} --update'.format(repo=REPO,
                                                                                          file_path=path,
                                                                                          destination=destination)
    p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = p.communicate(input=b'generated by automated airbnb knowledge repo setup')

    print('File: {path}'.format(path=os.path.split(path)[1]))
    print(cmd)
    print('\n')


def test_converters():
    # .Rmd
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

    # .ipynb
    PATH = '/Users/eric/Documents/machine-learning/data_science_is_software/notebooks/data_science_is_software.ipynb'
    converter = IpynbConverter(PATH)
    converter.convert()
    converter.write(inplace=False)


def main():
    convert_all_posts('/Users/eric/Documents/machine-learning/')


if __name__ == "__main__":
    main()
