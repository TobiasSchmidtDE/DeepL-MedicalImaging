import json
import os
from pathlib import Path
import re
import sys

token = os.environ['TOKEN']
WIKI_URL = 'https://oauth2:' + token + \
    '@git.veios.cloud/idp1-radio/idp-radio-1.wiki.git'


def execute():
    try:
        basepath = Path(os.path.dirname(
            os.path.realpath(__file__))).parent.parent
        logfile_path = basepath / 'logs/unvalidated-experiment-log.json'
        if not os.path.isfile(logfile_path):
            sys.exit(0)

        # load logfile
        f = open(logfile_path, 'r')
        data = json.load(f)
        f.close()

        # get experiments
        unvalidated_experiments = data['experiments']

        if len(unvalidated_experiments) < 1:
            sys.exit(1)

        # create temporary directory
        tempdir = str(basepath) + '/temp'
        if not os.path.isdir(tempdir):
            os.system('mkdir ' + tempdir)

        # clone wiki repository
        if not os.path.isdir(tempdir + '/idp-radio-1.wiki'):
            os.system('cd ' + tempdir + '&& git clone ' + WIKI_URL)

        wiki_model_dir = tempdir + '/idp-radio-1.wiki/models'
        if not os.path.isdir(wiki_model_dir):
            os.system('mkdir ' + wiki_model_dir)

        # execute notebook for each unvalidated model
        notebook_path = str(
            basepath / 'src/tests/validate_model.ipynb')
        for exp in unvalidated_experiments:
            # set model data as env variable and execute notebook
            os.environ['EXP_DATA'] = str(json.dumps(exp))
            output_path = str(basepath / 'temp' / exp['id'])
            os.system('jupyter nbconvert --to markdown --execute ' +
                      notebook_path + ' --output ' + output_path)

            # load exectuted notebooks data
            f = open(output_path + '.md', 'r')
            content = f.read()
            f.close()

            # move images to wiki folder and change references in md file
            for f in os.listdir(tempdir):
                if f.split('.')[1] == 'png':
                    os.system('mv ' + tempdir + '/' + f + ' ' +
                              tempdir + '/idp-radio-1.wiki/uploads/' + f)

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
            f = open(wiki_model_dir + '/' + exp['name'] + '.md', mode)
            f.write(content)
            f.close()

            # if no model with same name exists add to home page of wiki
            if not mode == 'a':
                f = open(tempdir + '/idp-radio-1.wiki/home.md', 'a')
                f.write('\n - [' + exp['name'] + '](' +
                        'models/' + exp['name'] + ')')
                f.close()

            # commit and push changes to wiki git
            os.system('cd ' + tempdir + '/idp-radio-1.wiki; git add .; git commit -m "Add model ' +
                      exp['name'] + '"; git push;')

        # write to unvalidated experiments log file
        f = open(logfile_path, 'w')
        json_data = json.dumps({'experiments': []}, indent=4)
        f.write(json_data)
        f.close()

        # append unvalidated models to main log file
        main_logfile_path = basepath / 'logs/experiment-log.json'
        f = open(main_logfile_path, 'r')
        main_data = json.load(f)
        f.close()

        experiments = main_data['experiments']
        main_data['experiments'] = experiments + unvalidated_experiments

        # write data back to logfile
        f = open(main_logfile_path, 'w')
        json_data = json.dumps(main_data, indent=4)
        f.write(json_data)
        f.close()

        # remove temp dir
        os.system('rm -rf ' + tempdir)

    # pylint: disable=bare-except
    except:
        print('An error occurred', sys.exc_info()[0])
        sys.exit(0)


execute()
