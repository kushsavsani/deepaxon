import os
import pandas as pd
from morphometrics import get_morphometrics

img_dir = r"D:\Research\Isaacs Lab\JimDA Segmented Ran"
overall_morph = pd.DataFrame(columns=['Vial', 'Section', 'Axon Count', 'Avg G-Ratio'])
current_slide = '01'
slide_df = pd.DataFrame()
slide_morph = pd.DataFrame(columns=['Vial', 'Axon Count', 'Avg G-Ratio'])

for img_name in sorted(os.listdir(img_dir)):
    slide = img_name.split('.')[0].split('_')[2]
    sec = img_name.split('.')[0].split('_')[3]
    img_path = os.path.join(img_dir, img_name)
    
    img_morph = get_morphometrics(img_path)
    img_morph = img_morph[img_morph['myelin_thickness']>0]
    axon_count = len(img_morph)
    gratio = img_morph['gratio'].mean()
    
    img_df = pd.DataFrame({'Vial':[slide], 'Section':[sec], 'Axon Count':[axon_count], 'Avg G-Ratio':[gratio]})
    overall_morph = pd.concat([overall_morph, img_df], axis=0)
    
    if slide == current_slide:
        slide_df = pd.concat([slide_df, img_morph], axis=0)
    else:
        print(slide, current_slide)
        slide_row = pd.DataFrame({'Vial':[current_slide], 'Axon Count':[len(slide_df)], 'Avg G-Ratio':[slide_df['gratio'].mean()]})
        slide_morph = pd.concat([slide_morph, slide_row], axis=0)
        current_slide = slide
        slide_df = img_df

slide_morph.to_excel(r"D:\Research\Isaacs Lab\JimDA Segmented Ran\slide_morphometrics.xlsx")
overall_morph.to_excel(r"D:\Research\Isaacs Lab\JimDA Segmented Ran\section_morphometrics.xlsx")