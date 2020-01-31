# =================
# get data from wiki
# =================
import json
import os
import shutil
from pathlib import Path

url_list = ['https://en.wikipedia.org/wiki/Acne',
            'https://en.wikipedia.org/wiki/Alcoholism',
            'https://en.wikipedia.org/wiki/Alcohol_abuse',
            'https://en.wikipedia.org/wiki/Alzheimer%27s_disease',
            'https://en.wikipedia.org/wiki/Appendicitis',
            'https://en.wikipedia.org/wiki/Asperger_syndrome',
            'https://en.wikipedia.org/wiki/Asthma',
            'https://en.wikipedia.org/wiki/Attention_deficit_hyperactivity_disorder',
            'https://en.wikipedia.org/wiki/Autism',
            'https://en.wikipedia.org/wiki/Bipolar_disorder',
            'https://en.wikipedia.org/wiki/Breast_cancer',
            'https://en.wikipedia.org/wiki/Cancer',
            'https://en.wikipedia.org/wiki/Chikungunya',
            'https://en.wikipedia.org/wiki/Cholera',
            'https://en.wikipedia.org/wiki/Common_cold',
            'https://en.wikipedia.org/wiki/Dengue_fever',
            'https://en.wikipedia.org/wiki/Depression',
            'https://en.wikipedia.org/wiki/Depression_(mood)',
            'https://en.wikipedia.org/wiki/Dysthymia',
            'https://en.wikipedia.org/wiki/Major_depressive_disorder',
            'https://en.wikipedia.org/wiki/Diabetes',
            'https://en.wikipedia.org/wiki/Diphtheria',
            'https://en.wikipedia.org/wiki/Epilepsy',
            'https://en.wikipedia.org/wiki/Erectile_dysfunction',
            'https://en.wikipedia.org/wiki/Far-sightedness',
            'https://en.wikipedia.org/wiki/Candidiasis',
            'https://en.wikipedia.org/wiki/Genital_herpes',
            'https://en.wikipedia.org/wiki/Gestational_diabetes',
            'https://en.wikipedia.org/wiki/Gonorrhea',
            'https://en.wikipedia.org/wiki/Gout',
            'https://en.wikipedia.org/wiki/Heartburn',
            'https://en.wikipedia.org/wiki/Hepatitis_B',
            'https://en.wikipedia.org/wiki/Herpes_simplex',
            'https://en.wikipedia.org/wiki/HIV/AIDS',
            'https://en.wikipedia.org/wiki/Genital_wart',
            'https://en.wikipedia.org/wiki/Hypertension',
            'https://en.wikipedia.org/wiki/Insomnia',
            'https://en.wikipedia.org/wiki/Irritable_bowel_syndrome',
            'https://en.wikipedia.org/wiki/Jaundice',
            'https://en.wikipedia.org/wiki/Kawasaki_disease',
            'https://en.wikipedia.org/wiki/Leukemia',
            'https://en.wikipedia.org/wiki/Cirrhosis',
            'https://en.wikipedia.org/wiki/Hypotension',
            'https://en.wikipedia.org/wiki/Malaria',
            'https://en.wikipedia.org/wiki/Migraine',
            'https://en.wikipedia.org/wiki/Multiple_sclerosis',
            'https://en.wikipedia.org/wiki/Near-sightedness',
            'https://en.wikipedia.org/wiki/Non-gonococcal_urethritis',
            'https://en.wikipedia.org/wiki/Obesity',
            'https://en.wikipedia.org/wiki/Lichen_planus',
            'https://en.wikipedia.org/wiki/Osteoarthritis',
            'https://en.wikipedia.org/wiki/Ovarian_cancer',
            'https://en.wikipedia.org/wiki/Polycystic_ovary_syndrome',
            'https://en.wikipedia.org/wiki/Postpartum_depression',
            'https://en.wikipedia.org/wiki/Psoriasis',
            'https://en.wikipedia.org/wiki/Rabies',
            'https://en.wikipedia.org/wiki/Rheumatoid_arthritis',
            'https://en.wikipedia.org/wiki/Schizophrenia',
            'https://en.wikipedia.org/wiki/Stroke',
            'https://en.wikipedia.org/wiki/Swine_influenza',
            'https://en.wikipedia.org/wiki/Syphilis',
            'https://en.wikipedia.org/wiki/Trichomoniasis',
            'https://en.wikipedia.org/wiki/Tuberculosis',
            'https://en.wikipedia.org/wiki/Ulcer',
            'https://en.wikipedia.org/wiki/Ulcer_(dermatology)',
            'https://en.wikipedia.org/wiki/Pressure_ulcer',
            'https://en.wikipedia.org/wiki/Genital_ulcer',
            'https://en.wikipedia.org/wiki/Ulcerative_dermatitis',
            'https://en.wikipedia.org/wiki/Anal_fissure',
            'https://en.wikipedia.org/wiki/Diabetic_foot_ulcer',
            'https://en.wikipedia.org/wiki/Corneal_ulcer',
            'https://en.wikipedia.org/wiki/Mouth_ulcer',
            'https://en.wikipedia.org/wiki/Aphthous_stomatitis',
            'https://en.wikipedia.org/wiki/Peptic_ulcer_disease',
            'https://en.wikipedia.org/wiki/Venous_ulcer',
            'https://en.wikipedia.org/wiki/Stress_ulcer',
            'https://en.wikipedia.org/wiki/Skin_manifestations_of_sarcoidosis',
            'https://en.wikipedia.org/wiki/Lichen_planus',
            'https://en.wikipedia.org/wiki/Ulcerative_colitis',
            'https://en.wikipedia.org/wiki/Urinary_tract_infection',
            'https://en.wikipedia.org/wiki/Vaginitis',
            'https://en.wikipedia.org/wiki/Varicose_veins',
            'https://en.wikipedia.org/wiki/Granulomatosis_with_polyangiitis',
            'https://en.wikipedia.org/wiki/Vaginal_yeast_infection']



ill_list = ['Acne', 'Alcoholism / Alcohol Use Disorder','Alcoholism / Alcohol Use Disorder',
            "Alzheimer's Disease", 'Appendicitis', 'Asperger’s Syndrome', 'Asthma','Attention Deficit Hyperactivity Disorder',
            'Autism', 'Bipolar Disorder', 'Breast Cancer', 'Cancer ', 'Chikungunya', 'Cholera', 'Common cold and flu',
            'Dengue','Depression', 'Depression','Depression','Depression','Diabetes', 'Diphtheria', 'Epilepsy', 'Erectile Dysfunction',
            'Farsightedness (or Hyperopia)', 'Genital candidiasis', 'Genital Herpes', 'Gestational Diabetes', 'Gonorrhoea ', 'Gout',
            'Heartburn', 'Hepatitis B', 'Herpes', 'HIV-AIDS', 'HPV genital warts', 'Hypertension or high blood pressure', 'Insomnia',
            'Irritable bowel syndrome', 'Jaundice', 'Kawasaki Syndrome', 'Leukemia', 'Liver Cirrhosis', 'Low blood pressure',
            'Malaria', 'Migraine', 'Multiple Sclerosis', 'Nearsightedness (or Myopia)', 'Non-Gonococcal Urethritis (NGU)',
            'Obesity', 'Oral lichen planus', 'Osteoarthritis', 'Ovarian cancer', 'Poly-cystic Ovarian Disease or Syndrome (PCOD/PCOS)',
            'Prenatal Depression', 'Psoriasis', 'Rabies', 'Rheumatoid Arthritis', 'Schizophrenia', 'Stroke', 'Swine flu', 'Syphilis',
            'Trichomoniasis', 'Tuberculosis','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers',
            'Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Ulcers','Urinary Tract Infection', 'Vaginitis', 'Varicose Veins',
            "Wegener's Disease (Granulomatosis with polyangiitis)", 'Yeast Infection (Vaginal)']

map_value = ['acne', 'alcoholism-','alcoholism-', 'alzheimers-disease', 'appendicitis', 'aspergers-syndrome', 'asthma',
             'attention-deficit-hyperactivity-disorder', 'autism', 'bipolar-disorder', 'breast-cancer', 'cancer', 'chikungunya',
             'cholera', 'common-cold-and-flu', 'dengue', 'depression', 'depression', 'depression', 'depression', 'diabetes',
             'diphtheria', 'epilepsy', 'erectile-dysfunction', 'farsightedness-or-hyperopia', 'genital-candidiasis', 'genital-herpes',
             'gestational-diabetes', 'gonorrhoea', 'gout', 'heartburn', 'hepatitis-b', 'herpes', 'hiv', 'hpv-genital-warts',
             'hypertension-or-high-blood-pressure', 'insomnia', 'irritable-bowel-syndrome', 'jaundice', 'kawasaki-syndrome',
             'leukemia', 'liver-cirrhosis', 'low-blood-pressure', 'malaria', 'migraine', 'multiple-sclerosis',
             'nearsightedness-or-myopia', 'non-gonococcal-urethritis-ngu', 'obesity', 'oral-lichen-planus', 'osteoarthritis',
             'ovarian-cancer', 'poly-cystic-ovarian-disease-or-syndrome-pcod', 'prenatal-depression', 'psoriasis', 'rabies',
             'rheumatoid-arthritis', 'schizophrenia', 'stroke', 'swine-flu', 'syphilis', 'trichomoniasis', 'tuberculosis',
             'ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers','ulcers',
             'ulcers','ulcers','ulcers','ulcers','urinary-tract-infection', 'vaginitis', 'varicose-veins',
             'wegeners-disease-granulomatosis-with-polyangiitis', 'yeast-infection-vaginal']
import scrapy

map_key = [[x for x in ele.split("/") if x][3] for ele in url_list]
dire_dict = dict(zip(map_key, map_value))

class QuotesSpider(scrapy.Spider):
    start_urls = url_list

    def parse(self, response):
        temp = [x for x in response.url.split("/") if x]
        if len(temp)>3 and os.path.exists('./timesofindia/'+ dire_dict[temp[3]]):

            filename = './timesofindia/'+dire_dict[temp[3]] +"/wiki_" +"_".join(temp[3:]).split(".")[0]+ '.txt'
            content_list  = response.css('div.mw-body-content').xpath('.//p').xpath('.//text()').extract()
            with open(filename, 'a+') as f:
                 _ = [f.write(ele+'\n') for ele in content_list]

