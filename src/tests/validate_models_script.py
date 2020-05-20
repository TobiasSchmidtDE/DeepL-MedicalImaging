import json
import os
from pathlib import Path
import re
import sys
import traceback

token = os.environ['TOKEN']
WIKI_URL = 'https://oauth2:' + token + \
    '@git.veios.cloud/idp1-radio/idp-radio-1.wiki.git'


def execute():
    try:
        basepath = Path(os.getcwd())
        # make sure your working directory is the repository root.
        if basepath.name != "idp-radio-1":
            basepath = basepath.parent.parent

        logfile_path = basepath / "logs" / "unvalidated-experiment-log.json"

        if not logfile_path.is_file():
            sys.exit(0)

        # load logfile
        f = open(logfile_path, 'r')
        data = json.load(f)
        f.close()

        # get experiments
        unvalidated_experiments = data['experiments']

        if len(unvalidated_experiments) < 1:
            return

        # create temporary directory
        tempdir = basepath / 'temp'
        tempdir.mkdir(parents=True, exist_ok=True)

        # clone wiki repository
        wikidir = tempdir / 'idp-radio-1.wiki'
        if not wikidir.is_dir():
            os.system('cd ' + str(tempdir) + '&& git clone ' + str(WIKI_URL))
        else:
            os.system('cd ' + str(tempdir) + '&& git pull ')

        wiki_model_dir = wikidir / "models"
        wiki_model_dir.mkdir(parents=True, exist_ok=True)

        # execute notebook for each unvalidated model
        notebook_path = basepath / 'src/tests/validate_model.ipynb'
        for exp in unvalidated_experiments:
            # set model data as env variable and execute notebook
            os.environ['EXP_DATA'] = str(json.dumps(exp))
            output_path = tempdir / exp['id']
            os.system('jupyter nbconvert --to markdown --execute ' +
                      str(notebook_path) + ' --output ' + str(output_path))

            # load exectuted notebooks data
            f = open(output_path / '.md', 'r')
            content = f.read()
            f.close()

            # move images to wiki folder and change references in md file
            for f in os.listdir(tempdir):
                if f.split('.')[1] == 'png':
                    target = tempdir / f
                    destination = tempdir / "idp-radio-1.wiki" / "uploads" / f
                    os.system('mv ' + str(target) + ' ' + str(destination))

                    regex = r'!\[png\]\(\S+' + f + r'\)'
                    content = re.sub(
                        regex, '![png](../uploads/' + f + ')', content)

            # add heading to md file
            heading = '# ' + exp['name']
            heading += '\n'
            heading += 'Version: ' + exp['version']
            heading += '\n\n'
            heading += exp['description']
            heading += '\n\n'
            content = heading + content

            # write md file into wiki, either append if model with same name exists
            # or create a new file
            if os.path.exists(wiki_model_dir + '/' + exp['name'] + '.md'):
                mode = 'a'
            else:
                mode = 'w'

            with open(wiki_model_dir + '/' + exp['name'] + '.md', mode) as f:
                f.write(content)

            # if no model with same name exists add to home page of wiki
            if not mode == 'a':
                with open(tempdir + '/idp-radio-1.wiki/home.md', 'a') as f:
                    f.write('\n - [' + exp['name'] + '](' +
                            'models/' + exp['name'] + ')')

            # commit and push changes to wiki git
            commit_msg = "Add model '" + exp['name'] + "'"
            os.system('cd ' + str(wikidir) +
                      '; git add .; git commit -m ' + str(commit_msg) + '; git push;')

        # write to unvalidated experiments log file
        with open(logfile_path, 'w') as f:
            json_data = json.dumps({'experiments': []}, indent=4)
            f.write(json_data)

        # append unvalidated models to main log file
        main_logfile_path = basepath / 'logs/experiment-log.json'
        with open(main_logfile_path, 'r') as f:
            main_data = json.load(f)

        experiments = main_data['experiments']
        main_data['experiments'] = experiments + unvalidated_experiments

        # write data back to logfile
        with open(main_logfile_path, 'w') as f:
            json_data = json.dumps(main_data, indent=4)
            f.write(json_data)

        # remove temp dir
        os.system('rm -rf ' + str(tempdir))

    # pylint: disable=bare-except
    except:
        traceback.print_exc()
        sys.exit(1)


execute()
