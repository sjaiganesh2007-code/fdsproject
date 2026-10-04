import os
import random
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Starting data generation for AutoBill Verify...")
    
    # Ensure directories exist
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # -------------------------------------------------------------
    # 1. BUILD CGHS RATE LIST (3,000+ items)
    # -------------------------------------------------------------
    cghs_records = []
    item_id_counter = 1

    # Category 1: Consultations
    specialties = [
        "General Medicine", "Cardiology", "Neurology", "Orthopedics", "Oncology",
        "Pediatrics", "Nephrology", "Gastroenterology", "Pulmonology", "Dermatology",
        "ENT", "Ophthalmology", "General Surgery", "Urology", "Gynecology",
        "Rheumatology", "Psychiatry", "Neurosurgery", "Cardiothoracic Surgery", "Endocrinology"
    ]
    consult_types = [
        ("OPD Routine Consultation", 300, 35),
        ("OPD Specialist Consultation", 500, 50),
        ("OPD Super Specialist Consultation", 800, 80),
        ("IPD Specialist Visit per day", 600, 60),
        ("IPD ICU Consultant Visit", 1200, 150),
        ("Emergency Triage Consultation", 750, 75),
        ("Senior Consultant Review", 900, 90),
        ("Teleconsultation Session", 400, 40),
        ("Pre-Operative Evaluation", 650, 65),
        ("Post-Operative Followup", 350, 35)
    ]
    modifiers = ["Standard", "Follow-Up", "Emergency Hours", "VIP Care", "Weekend Special"]

    for spec in specialties:
        for ctype, base_rate, base_std in consult_types:
            for mod in modifiers:
                rate = round(base_rate * (1.0 + random.uniform(-0.15, 0.25)), 2)
                std_dev = round(max(5.0, base_std * random.uniform(0.8, 1.2)), 2)
                item_name = f"{mod} {ctype} - {spec}"
                cghs_records.append({
                    "item_code": f"CGHS_CONS_{item_id_counter:05d}",
                    "item_name": item_name,
                    "category": "Consultation",
                    "cghs_rate": rate,
                    "std_dev": std_dev
                })
                item_id_counter += 1

    # Category 2: ICU & Room Rent
    bed_types = [
        ("General Ward Non-AC Bed per day", 1000, 100),
        ("General Ward AC Bed per day", 1500, 150),
        ("Semi-Private Room Bed per day", 2500, 250),
        ("Private Room Single AC Bed per day", 4500, 450),
        ("Deluxe Room Bed per day", 6500, 650),
        ("Super Deluxe Room Bed per day", 9000, 900),
        ("Suite Room Bed per day", 12000, 1200),
        ("High Dependency Unit (HDU Bed per day)", 5000, 500),
        ("Intensive Care Unit (ICU Bed per day)", 8000, 800),
        ("Intensive Coronary Care Unit (ICCU Bed per day)", 8500, 850),
        ("Pediatric ICU (PICU Bed per day)", 7500, 750),
        ("Neonatal ICU Level 1 Bed per day", 6000, 600),
        ("Neonatal ICU Level 2 Bed per day", 8000, 800),
        ("Neonatal ICU Level 3 Bed per day", 10000, 1000),
        ("Surgical ICU Bed per day", 8500, 850),
        ("Neuro ICU Bed per day", 9500, 950),
        ("Isolation ICU Bed per day", 9000, 900),
        ("Burn Unit ICU Bed per day", 8500, 850),
        ("Day Care Bed per day", 2000, 200)
    ]
    room_tiers = ["Tier-1 City", "Tier-2 City", "NABH Accredited", "Non-NABH Accredited", "Super-Specialty Wing"]
    for bname, base_rate, base_std in bed_types:
        for rtier in room_tiers:
            rate = round(base_rate * (1.0 + random.uniform(-0.1, 0.2)), 2)
            std_dev = round(max(10.0, base_std * random.uniform(0.85, 1.15)), 2)
            item_name = f"{bname} ({rtier})"
            cghs_records.append({
                "item_code": f"CGHS_ROOM_{item_id_counter:05d}",
                "item_name": item_name,
                "category": "Room Rent",
                "cghs_rate": rate,
                "std_dev": std_dev
            })
            item_id_counter += 1

    # Category 3: Pathology Diagnostics
    pathology_tests = [
        ("Complete Blood Count (CBC)", 250, 25), ("Hemoglobin (Hb)", 90, 10),
        ("Total Leucocyte Count (TLC)", 100, 10), ("Differential Leucocyte Count (DLC)", 100, 10),
        ("Platelet Count", 120, 12), ("Erythrocyte Sedimentation Rate (ESR)", 80, 8),
        ("Blood Grouping & Rh Typing", 110, 11), ("Packed Cell Volume (PCV)", 90, 9),
        ("Reticulocyte Count", 150, 15), ("Peripheral Blood Smear", 150, 15),
        ("Fasting Blood Sugar (FBS)", 80, 8), ("Postprandial Blood Sugar (PPBS)", 80, 8),
        ("Random Blood Sugar (RBS)", 70, 7), ("HbA1c Glycated Hemoglobin", 450, 45),
        ("Oral Glucose Tolerance Test (OGTT)", 250, 25), ("Total Cholesterol", 150, 15),
        ("HDL Cholesterol", 160, 16), ("LDL Cholesterol", 180, 18),
        ("Triglycerides", 170, 17), ("VLDL Cholesterol", 140, 14),
        ("Lipid Profile Complete", 550, 55), ("Serum Creatinine", 120, 12),
        ("Blood Urea Nitrogen (BUN)", 130, 13), ("Uric Acid", 140, 14),
        ("Serum Sodium (Na+)", 110, 11), ("Serum Potassium (K+)", 110, 11),
        ("Serum Chloride (Cl-)", 110, 11), ("Serum Calcium", 150, 15),
        ("Serum Phosphorus", 150, 15), ("Serum Magnesium", 200, 20),
        ("Kidney Function Test (KFT)", 600, 60), ("Liver Function Test (LFT)", 650, 65),
        ("Serum Bilirubin Total & Direct", 180, 18), ("SGOT / AST", 130, 13),
        ("SGPT / ALT", 130, 13), ("Alkaline Phosphatase (ALP)", 140, 14),
        ("Serum Albumin", 120, 12), ("Serum Globulin", 120, 12),
        ("Total Serum Protein", 130, 13), ("Thyroid Profile (T3 T4 TSH)", 500, 50),
        ("Free T3 (FT3)", 220, 22), ("Free T4 (FT4)", 220, 22),
        ("Anti-TPO Antibodies", 800, 80), ("Vitamin D (25-OH)", 1100, 110),
        ("Vitamin B12", 850, 85), ("Serum Ferritin", 600, 60),
        ("Serum Iron Profile", 700, 70), ("Total Iron Binding Capacity (TIBC)", 300, 30),
        ("Dengue NS1 Antigen", 600, 60), ("Dengue IgG/IgM Antibodies", 650, 65),
        ("Chikungunya PCR Test", 1200, 120), ("Malaria Rapid Antigen (Pf/Pv)", 300, 30),
        ("Typhoid Widal Test", 200, 20), ("Typhoid Typhidot Test", 400, 40),
        ("Urine Routine & Microscopy", 100, 10), ("Urine Microalbumin", 350, 35),
        ("Urine Protein 24 Hours", 250, 25), ("Urine Culture & Sensitivity", 450, 45),
        ("Blood Culture & Sensitivity", 750, 75), ("Sputum AFB Stain for TB", 150, 15),
        ("Sputum Culture", 450, 45), ("Stool Routine & Microscopy", 120, 12),
        ("Stool Occult Blood Test", 150, 15), ("C-Reactive Protein (CRP) Quantitative", 400, 40),
        ("High Sensitivity CRP (hs-CRP)", 600, 60), ("D-Dimer Quantitative", 1100, 110),
        ("Prothrombin Time (PT/INR)", 250, 25), ("Activated Partial Thromboplastin Time (APTT)", 300, 30),
        ("Fibrinogen Level", 450, 45), ("Troponin I Quantitative", 900, 90),
        ("Troponin T Quantitative", 1000, 100), ("NT-proBNP Marker", 2200, 220),
        ("Procalcitonin (PCT)", 2500, 250), ("Serum Amylase", 350, 35),
        ("Serum Lipase", 450, 45), ("Lactate Dehydrogenase (LDH)", 300, 30),
        ("CPK-MB", 400, 40), ("CPK Total", 350, 35),
        ("HBsAg Hepatitis B Surface Antigen", 300, 30), ("Anti-HCV Antibody", 400, 40),
        ("HIV 1 & 2 ELISA Screen", 350, 35), ("VDRL / RPR Syphilis Screen", 150, 15),
        ("Prostate Specific Antigen (PSA Total)", 650, 65), ("CA-125 Ovarian Tumor Marker", 950, 95),
        ("CEA Carcinoembryonic Antigen", 850, 85), ("Alpha Fetoprotein (AFP)", 750, 75),
        ("Beta-hCG Quantitative", 600, 60), ("PAP Smear Cytology", 400, 40),
        ("Arterial Blood Gas (ABG)", 500, 50), ("Serum Electrolytes Panel", 300, 30)
    ]
    path_methods = [
        "Automated Analyzer", "ELISA Technique", "CLIA Method",
        "Rapid Card Screen", "Real-Time PCR", "Manual Spectrophotometry",
        "HPLC Method", "Fluorescence Immunoassay", "Routine Protocol"
    ]
    for ptest, base_rate, base_std in pathology_tests:
        for pmethod in path_methods:
            rate = round(base_rate * (1.0 + random.uniform(-0.1, 0.15)), 2)
            std_dev = round(max(3.0, base_std * random.uniform(0.8, 1.2)), 2)
            item_name = f"{ptest} - {pmethod}"
            cghs_records.append({
                "item_code": f"CGHS_PATH_{item_id_counter:05d}",
                "item_name": item_name,
                "category": "Diagnostics - Pathology",
                "cghs_rate": rate,
                "std_dev": std_dev
            })
            item_id_counter += 1

    # Category 4: Imaging Diagnostics (CT Scan, MRI, Ultrasound, X-Ray)
    imaging_tests = [
        ("CT Scan Head Plain", 1800, 180), ("CT Scan Head Contrast", 2500, 250),
        ("HRCT Scan Chest", 3200, 320), ("CT Scan Abdomen & Pelvis Plain", 3500, 350),
        ("CT Scan Abdomen & Pelvis Contrast", 4800, 480), ("CT Angiography Coronary", 7500, 750),
        ("CT Angiography Brain", 6000, 600), ("CT Scan Spine Cervical", 2800, 280),
        ("CT Scan Spine Lumbar", 2800, 280), ("CT Paranasal Sinuses (PNS)", 2200, 220),
        ("CT Temporal Bone", 2500, 250), ("CT Scan KUB", 3000, 300),
        ("MRI Brain Plain", 3200, 320), ("MRI Brain Contrast", 4800, 480),
        ("MRI Spine Cervical", 3500, 350), ("MRI Spine Thoracic", 3500, 350),
        ("MRI Spine Lumbar", 3500, 350), ("MRI Knee Joint", 3800, 380),
        ("MRI Shoulder Joint", 3800, 380), ("MRI Hip Joint", 3800, 380),
        ("MRI Whole Abdomen", 5500, 550), ("MRI Pelvis", 4500, 450),
        ("MRCP Biliary Tree", 4200, 420), ("MR Angiography Brain", 5000, 500),
        ("USG Whole Abdomen", 800, 80), ("USG Upper Abdomen", 600, 60),
        ("USG Pelvis & Lower Abdomen", 600, 60), ("USG KUB", 650, 65),
        ("USG Obstetrics Anomaly Level-2", 1200, 120), ("USG Doppler Lower Limb Arterial", 1800, 180),
        ("USG Doppler Lower Limb Venous", 1800, 180), ("USG Doppler Carotid", 1600, 160),
        ("USG Scrotum & Testes", 900, 90), ("USG Neck & Thyroid", 850, 85),
        ("Mammography Bilateral", 1400, 140), ("X-Ray Chest PA View", 250, 25),
        ("X-Ray Spine Cervical AP/Lat", 400, 40), ("X-Ray Spine Lumbar AP/Lat", 400, 40),
        ("X-Ray Knee Joint AP/Lat", 350, 35), ("X-Ray Hip Joint AP/Lat", 350, 35),
        ("X-Ray Abdomen Erect", 300, 30), ("ECG 12 Lead Tracing", 150, 15),
        ("2D Echocardiogram with Color Doppler", 1500, 150), ("Stress Echocardiogram", 2500, 250),
        ("Treadmill Exercise Test (TMT)", 1100, 110), ("Holter Monitoring 24 Hours", 2200, 220),
        ("EEG Routine Sleep & Awake", 1200, 120), ("EMG Nerve Conduction Velocity (NCV)", 1800, 180),
        ("Pulmonary Function Test (PFT Spirometry)", 700, 70), ("PET-CT Whole Body Scan", 14000, 1400),
        ("DEXA Bone Densitometry Whole Body", 1800, 180)
    ]
    img_variations = [
        "Standard Protocol", "High Resolution", "With Contrast Media",
        "3D Reconstruction", "Multi-Slice 64", "1.5 Tesla MRI",
        "3.0 Tesla MRI", "Emergency Express", "Digital Radiography"
    ]
    for itest, base_rate, base_std in imaging_tests:
        for ivar in img_variations:
            rate = round(base_rate * (1.0 + random.uniform(-0.1, 0.2)), 2)
            std_dev = round(max(5.0, base_std * random.uniform(0.8, 1.2)), 2)
            item_name = f"{itest} ({ivar})"
            cghs_records.append({
                "item_code": f"CGHS_IMG_{item_id_counter:05d}",
                "item_name": item_name,
                "category": "Diagnostics - Imaging",
                "cghs_rate": rate,
                "std_dev": std_dev
            })
            item_id_counter += 1

    # Category 5: Surgeries & Procedures
    surgeries = [
        ("Cataract Surgery Phacoemulsification with Foldable IOL", 15000, 1500),
        ("Cataract Surgery MICS Laser", 22000, 2200),
        ("Laparoscopic Cholecystectomy", 28000, 2800),
        ("Open Cholecystectomy", 22000, 2200),
        ("Laparoscopic Appendectomy", 25000, 2500),
        ("Open Appendectomy", 18000, 1800),
        ("Inguinal Hernia Repair Laparoscopic Mesh", 30000, 3000),
        ("Inguinal Hernia Repair Open Mesh", 22000, 2200),
        ("Ventral Hernia Repair Laparoscopic", 32000, 3200),
        ("Hemorrhoidectomy Stapled Laser", 24000, 2400),
        ("Anal Fissurectomy & Sphincterotomy", 16000, 1600),
        ("Fistulotomy / Fistulectomy", 18000, 1800),
        ("Hydrocelectomy Unilateral", 14000, 1400),
        ("Cesarean Section C-Section Emergency", 28000, 2800),
        ("Cesarean Section C-Section Elective", 24000, 2400),
        ("Normal Vaginal Delivery", 16000, 1600),
        ("Total Laparoscopic Hysterectomy (TLH)", 40000, 4000),
        ("Total Abdominal Hysterectomy (TAH)", 32000, 3200),
        ("Vaginal Hysterectomy with Pelvic Repair", 35000, 3500),
        ("Laparoscopic Ovarian Cystectomy", 28000, 2800),
        ("Myomectomy Laparoscopic", 34000, 3400),
        ("Coronary Artery Bypass Grafting (CABG Off-Pump)", 130000, 13000),
        ("Coronary Artery Bypass Grafting (CABG On-Pump)", 145000, 14500),
        ("Single Vessel Coronary Stenting (PCI)", 65000, 6500),
        ("Double Vessel Coronary Stenting (PCI)", 95000, 9500),
        ("Triple Vessel Coronary Stenting (PCI)", 125000, 12500),
        ("Pacemaker Implantation Single Chamber", 45000, 4500),
        ("Pacemaker Implantation Dual Chamber", 75000, 7500),
        ("Total Knee Replacement (TKR Unilateral)", 110000, 11000),
        ("Total Knee Replacement (TKR Bilateral)", 190000, 19000),
        ("Total Hip Replacement (THR Unilateral)", 120000, 12000),
        ("Total Hip Replacement (THR Bilateral)", 210000, 21000),
        ("Arthroscopic ACL Reconstruction Knee", 45000, 4500),
        ("Arthroscopic Menisctomy Knee", 32000, 3200),
        ("Transurethral Resection of Prostate (TURP)", 28000, 2800),
        ("TURBT Bladder Tumor Resection", 26000, 2600),
        ("Ureteroscopy URS with Laser Lithotripsy", 30000, 3000),
        ("Retrograde Intrarenal Surgery (RIRS)", 42000, 4200),
        ("Percutaneous Nephrolithotomy (PCNL)", 38000, 3800),
        ("AV Fistula Creation Surgery", 15000, 1500),
        ("Tympanoplasty Ear Surgery", 22000, 2200),
        ("Septoplasty Nasal Surgery", 18000, 1800),
        ("Functional Endoscopic Sinus Surgery (FESS)", 26000, 2600),
        ("Tonsillectomy with Adenoidectomy", 18000, 1800),
        ("Mastoidectomy Radical", 30000, 3000),
        ("Upper GI Endoscopy Diagnostic", 3500, 350),
        ("Colonoscopy Diagnostic Complete", 6000, 600),
        ("Flexible Bronchoscopy Procedure", 5500, 550),
        ("Hemodialysis Single Session", 1800, 180),
        ("Lumbar Puncture Diagnostic Procedure", 2200, 220),
        ("Central Venous Line Placement", 3500, 350),
        ("Arterial Line Insertion Procedure", 3000, 300),
        ("Intercostal Drain (ICD) Tube Insertion", 4000, 400),
        ("Percutaneous Bedside Tracheostomy", 8000, 800),
        ("Wound Debridement Surgical Large", 6000, 600),
        ("Wound Debridement Surgical Small", 2500, 250),
        ("Excision of Lipoma / Sebaceous Cyst", 4500, 450),
        ("Incision and Drainage (I&D) Abscess", 3000, 300)
    ]
    surg_grades = [
        "Uncomplicated", "Grade-1 Complexity", "Grade-2 Complexity",
        "Grade-3 Major", "Emergency Intervention", "Daycare Procedure",
        "NABH Standard Package", "Super Specialty Unit"
    ]
    for sname, base_rate, base_std in surgeries:
        for sgrade in surg_grades:
            rate = round(base_rate * (1.0 + random.uniform(-0.1, 0.2)), 2)
            std_dev = round(max(50.0, base_std * random.uniform(0.8, 1.2)), 2)
            item_name = f"{sname} - {sgrade}"
            cghs_records.append({
                "item_code": f"CGHS_SURG_{item_id_counter:05d}",
                "item_name": item_name,
                "category": "Surgeries & Procedures",
                "cghs_rate": rate,
                "std_dev": std_dev
            })
            item_id_counter += 1

    # Category 6: Common Indian Brand Medicines & Consumables
    medicines = [
        ("Dolo 650mg Tablet", 32, 3), ("Dolo 500mg Tablet", 25, 2.5),
        ("Dolo Suspension 60ml", 45, 4.5), ("Corex T Cough Syrup 100ml", 125, 12),
        ("Crocin 500mg Tablet", 28, 2.8), ("Crocin 650mg Advance Tablet", 35, 3.5),
        ("Crocin Pain Relief Tablet", 40, 4), ("Pantocid 40mg Tablet", 140, 14),
        ("Pantocid IV 40mg Injection", 55, 5.5), ("Augmentin 625mg Duo Tablet", 210, 20),
        ("Augmentin 1.2g IV Injection", 160, 16), ("Azithral 500mg Tablet", 120, 12),
        ("Azithral 250mg Tablet", 70, 7), ("Pan D Capsule", 150, 15),
        ("Ondem 4mg Tablet", 50, 5), ("Ondem 4mg/2ml Injection", 30, 3),
        ("Taxim O 200mg Tablet", 110, 11), ("Limcee 500mg Chewable Tablet", 25, 2.5),
        ("Becosules Z Capsule", 45, 4.5), ("Calpol 650mg Tablet", 30, 3),
        ("Calpol 500mg Tablet", 22, 2.2), ("Allegra 120mg Tablet", 190, 18),
        ("Allegra 180mg Tablet", 240, 24), ("Montair LC Tablet", 170, 17),
        ("Rabeprazole 20mg Tablet", 95, 9.5), ("Atorva 10mg Tablet", 110, 11),
        ("Atorva 20mg Tablet", 180, 18), ("Metformin 500mg SR Tablet", 35, 3.5),
        ("Metformin 1000mg SR Tablet", 60, 6), ("Telma 40mg Tablet", 130, 13),
        ("Telma H Tablet", 160, 16), ("Clopivas 75mg Tablet", 115, 11.5),
        ("Ecosprin 75mg Tablet", 15, 1.5), ("Ecosprin 150mg Tablet", 20, 2),
        ("Cilacar 10mg Tablet", 105, 10.5), ("Glycomet GP 2 Tablet", 140, 14),
        ("Amlovas 5mg Tablet", 35, 3.5), ("Crestor 10mg Tablet", 280, 28),
        ("Januvia 100mg Tablet", 420, 40), ("Forxiga 10mg Tablet", 560, 55),
        ("Lantus SoloStar Insulin Cartridge 100IU", 750, 75), ("Novorapid Flexpen 100IU/ml", 680, 65),
        ("Meropenem 1g IV Injection", 950, 90), ("Piptaz 4.5g IV Injection (Piperacillin+Tazobactam)", 480, 45),
        ("Ceftriaxone 1g IV Injection", 65, 6.5), ("Amikacin 500mg Injection", 85, 8.5),
        ("Vancomycin 1g Injection", 380, 38), ("Linezolid 600mg Tablet", 320, 30),
        ("Faropenem 200mg Tablet", 590, 55), ("Enoxaparin 40mg Clexane Injection", 550, 50),
        ("Heparin 5000 IU Injection", 140, 14), ("Tramadol 50mg Injection", 35, 3.5),
        ("Paracetamol 100ml IV Infusion Bottle", 75, 7.5), ("Dynapar AQ 75mg Injection", 30, 3),
        ("Emset 4mg Tablet", 45, 4.5), ("Deriphyllin 2ml Injection", 15, 1.5),
        ("Hydrocortisone 100mg Injection", 45, 4.5), ("Dexamethasone 8mg/2ml Injection", 20, 2),
        ("Pantoprazole 40mg IV Injection", 50, 5), ("Human Albumin 20% 100ml Infusion", 4200, 400),
        ("Noradrenaline 2mg/2ml Injection", 85, 8.5), ("Dobutamine 250mg/5ml Injection", 160, 16),
        ("Dopamine 200mg/5ml Injection", 75, 7.5), ("Ciprofloxacin 500mg Tablet", 65, 6.5),
        ("Levofloxacin 500mg Tablet", 90, 9), ("Clarithromycin 500mg Tablet", 260, 25),
        ("Metronidazole 400mg Tablet", 20, 2), ("Metronidazole 100ml IV Infusion", 40, 4),
        ("IV Normal Saline (NS 0.9%) 500ml", 45, 4.5), ("IV Ringer Lactate (RL) 500ml", 50, 5),
        ("IV Dextrose 5% 500ml", 45, 4.5), ("IV Dextrose Normal Saline (DNS) 500ml", 48, 4.8),
        ("Disposable Syringe 2ml with Needle", 8, 0.8), ("Disposable Syringe 5ml with Needle", 10, 1.0),
        ("Disposable Syringe 10ml with Needle", 14, 1.4), ("Disposable Syringe 50ml Luer Lock", 45, 4.5),
        ("IV Cannula 20G Pink", 40, 4), ("IV Cannula 18G Green", 40, 4),
        ("IV Cannula 22G Blue", 40, 4), ("Surgical Gloves Sterile Pair", 35, 3.5),
        ("N95 Respirator Mask", 60, 6), ("Surgical 3-Ply Face Mask", 10, 1.0),
        ("IV Infusion Set Vented", 30, 3), ("Microdrip IV Infusion Set", 55, 5.5),
        ("Urine Collecting Bag 2000ml", 65, 6.5), ("Foley Catheter 2-Way 16 Fr", 120, 12),
        ("Ryle's Nasogastric Tube 14 Fr", 75, 7.5), ("Suction Catheter 12 Fr", 35, 3.5),
        ("Endotracheal Tube Cuffed 7.5mm", 160, 16), ("Oxygen Mask with Tubing Adult", 90, 9),
        ("Venturi Mask with Kit", 180, 18), ("Nebulizer Mask Adult", 110, 11),
        ("Surgical Bandage Roll 4 inch", 25, 2.5), ("Surgical Bandage Roll 6 inch", 35, 3.5),
        ("Sterile Gauze Pad 10x10 cm Pack", 30, 3), ("Surgical Cotton Wool 500g", 140, 14),
        ("Surgical Blade No. 11", 12, 1.2), ("Alcohol Swab Prep Pad Box", 80, 8),
        ("Tegaderm Transparent Dressing 10x12 cm", 95, 9.5), ("Micropore Tape 1 inch Roll", 45, 4.5),
        ("Betadine Antiseptic Solution 500ml", 220, 20), ("Hand Sanitizer Gel 500ml", 180, 18),
        ("Disposable PPE Kit Complete", 450, 45), ("Disposable Bed Sheet Non-Woven", 60, 6),
        ("Adult Incontinence Diaper L", 55, 5.5), ("Central Venous Catheter Triple Lumen Kit", 1800, 180)
    ]
    med_packs = [
        "Strip of 10 Tablets", "Strip of 15 Tablets", "Single Ampoule 2ml",
        "Single Vial 10ml", "Bottle 100ml", "Single Unit Pack", "Box of 10 Units"
    ]
    for mname, base_rate, base_std in medicines:
        for mpack in med_packs:
            rate = round(base_rate * (1.0 + random.uniform(-0.1, 0.15)), 2)
            std_dev = round(max(0.5, base_std * random.uniform(0.8, 1.2)), 2)
            item_name = f"{mname} ({mpack})"
            cghs_records.append({
                "item_code": f"CGHS_MED_{item_id_counter:05d}",
                "item_name": item_name,
                "category": "Medicines & Consumables",
                "cghs_rate": rate,
                "std_dev": std_dev
            })
            item_id_counter += 1

    cghs_df = pd.DataFrame(cghs_records)
    cghs_file = data_dir / "cghs_rate_list.csv"
    cghs_df.to_csv(cghs_file, index=False)
    print(f"Saved {len(cghs_df)} CGHS rate items to '{cghs_file}'.")

    # -------------------------------------------------------------
    # 2. BUILD BILLING DATASET (20,000+ line items across 120 hospitals)
    # -------------------------------------------------------------
    num_hospitals = 120
    hospitals = [f"HOSP_{i:03d}" for i in range(1, num_hospitals + 1)]
    
    total_line_items = 22000
    billing_records = []

    # Distribution probabilities: 70% normal, 20% moderate overcharge, 10% severe fraud
    normal_count = int(total_line_items * 0.70)
    mod_count = int(total_line_items * 0.20)
    fraud_count = total_line_items - normal_count - mod_count
    
    label_pool = (['normal'] * normal_count) + (['moderate'] * mod_count) + (['severe_fraud'] * fraud_count)
    random.shuffle(label_pool)

    for i in range(total_line_items):
        bill_id = f"BILL_{i // 5 + 1:06d}"  # ~5 items per bill
        hosp_id = random.choice(hospitals)
        
        # Pick a random item from cghs_df
        cghs_row = cghs_df.sample(n=1).iloc[0]
        cghs_rate = float(cghs_row['cghs_rate'])
        cghs_name = cghs_row['item_name']
        category = cghs_row['category']

        label = label_pool[i]
        
        if label == 'normal':
            # Billed price close to CGHS rate (0.92x to 1.18x)
            billed_price = cghs_rate * random.uniform(0.92, 1.18)
        elif label == 'moderate':
            # Overcharged 1.35x to 2.4x
            billed_price = cghs_rate * random.uniform(1.35, 2.40)
        else:
            # Severe fraud anomalies 3.5x to 12.0x
            billed_price = cghs_rate * random.uniform(3.50, 12.00)
            
        billed_price = round(max(1.0, billed_price), 2)
        quantity = random.choices([1, 2, 3, 5, 10], weights=[70, 15, 8, 4, 3])[0]
        
        # Billed item name: 65% exact, 35% realistic variations
        var_type = random.random()
        if var_type < 0.65:
            billed_item_name = cghs_name
        elif var_type < 0.80:
            billed_item_name = cghs_name.upper()
        elif var_type < 0.90:
            billed_item_name = cghs_name.lower()
        elif var_type < 0.95:
            billed_item_name = f"HOSP CHARGE: {cghs_name}"
        else:
            # Rearrange words slightly
            words = cghs_name.split()
            if len(words) > 2:
                billed_item_name = " ".join([words[-1]] + words[:-1])
            else:
                billed_item_name = cghs_name

        billing_records.append({
            "bill_id": bill_id,
            "hospital_id": hosp_id,
            "billed_item_name": billed_item_name,
            "billed_price": billed_price,
            "quantity": quantity,
            "total_billed_amount": round(billed_price * quantity, 2),
            "category": category,
            "ground_truth_category": label
        })

    billing_df = pd.DataFrame(billing_records)
    billing_file = data_dir / "billing_dataset.csv"
    billing_df.to_csv(billing_file, index=False)
    print(f"Saved {len(billing_df)} billing line items across {num_hospitals} hospitals to '{billing_file}'.")
    print("Data generation complete!")

if __name__ == "__main__":
    main()
