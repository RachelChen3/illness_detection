#!/usr/bin/env bash
pip -r install requirement.txt
cd data
scrapy runspider get_illness_names.py -o timesofindia.json
commendline: scrapy runspider get_sublinks.py -o sublink.json
scrapy runspider get_subpages.py
scrapy runspider get_wikis.py
python create_dataset.py
cd ..
python run_detection.py --do_train --no_cuda --evaluate_during_training
python run_detection.py --do_eval
