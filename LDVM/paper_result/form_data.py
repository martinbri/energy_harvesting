import pandas as pd
import numpy as np

def generate_form_data(num_rows=1000):
    data=np.loadtxt('data_Oscillating_foils.dat',delimiter=',')
    main_cols = ['C_T', 'C_p', '\eta']
    sub_cols = ['St_model', 'y_model', 'St_experiment', 'y_experiment','St_analytical', 'y_analytical']
    column_tuples = [(main, sub) for main in main_cols for sub in sub_cols]
    columns = pd.MultiIndex.from_tuples(column_tuples)


    data_ct=data[0:18]
    data_ct_model=data_ct[0:6]
    data_ct_experiment=data_ct[6:15]
    data_ct_analytical=data_ct[15:18]


    data_cp_model = data[18:24]
    print(data_cp_model)
    data_cp_experiment = data[24:33]
    data_cp_analytical = data[33:37]

    data_eta_model=data[37:43]
    data_eta_experiment=data[43:52]
    data_eta_analytical=data[52:58]





    arr = np.full((9, 18), np.nan)
    arr[0:6, 0:2] = data_ct_model
    arr[0:3, 4:6] = data_ct_analytical
    arr[0:9, 2:4] = data_ct_experiment
    arr[0:6, 6:8] = data_cp_model
    arr[0:4, 10:12] = data_cp_analytical
    arr[0:9, 8:10] = data_cp_experiment
    arr[0:6, 12:14] = data_eta_model
    arr[0:6, 16:18] = data_eta_analytical
    arr[0:9, 14:16] = data_eta_experiment
    
    df = pd.DataFrame(arr, columns=columns)

    df.to_csv('form_data_h0_0.25_alpha0_5_deg_phase_90_degree.csv', index=False)

    print(df)


    data_ct_model=data[58:64]
    data_ct_experiment=data[64:74]
    data_ct_analytical=data[74:78]
    print(len(data_ct_model))
    print(len(data_ct_experiment))
    print(len(data_ct_analytical))



    data_cp_model = data[78:84]
    data_cp_experiment = data[84:94]
    data_cp_analytical = data[94:99]

    print(len(data_cp_model))
    print(len(data_cp_experiment))
    print(len(data_cp_analytical))

    data_eta_model=data[99:105]
    data_eta_experiment=data[105:116]
    data_eta_analytical=data[116:120]
    print(len(data_eta_model))
    print(len(data_eta_experiment))
    print(len(data_eta_analytical))




    arr = np.full((11, 18), np.nan)
    arr[0:6, 0:2] = data_ct_model
    arr[0:4, 4:6] = data_ct_analytical
    arr[0:10, 2:4] = data_ct_experiment
    arr[0:6, 6:8] = data_cp_model
    arr[0:5, 10:12] = data_cp_analytical
    arr[0:10, 8:10] = data_cp_experiment
    arr[0:6, 12:14] = data_eta_model
    arr[0:4, 16:18] = data_eta_analytical
    arr[0:11, 14:16] = data_eta_experiment
    
    df = pd.DataFrame(arr, columns=columns)
    print(df)

    df.to_csv('form_data_h0_0.25_alpha0_15_deg_phase_90_degree.csv', index=False)

    data_ct_model=data[120:126]
    data_ct_experiment=data[126:136]
    data_ct_analytical=data[136:140]
    print(len(data_ct_model))
    print(len(data_ct_experiment))
    print(len(data_ct_analytical))



    data_cp_model = data[140:146]
    data_cp_experiment = data[146:156]
    data_cp_analytical = data[156:165]

    print(len(data_cp_model))
    print(len(data_cp_experiment))
    print(len(data_cp_analytical))

    data_eta_model=data[165:171]
    data_eta_experiment=data[171:181]
    data_eta_analytical=data[181:187]
    print(len(data_eta_model))
    print(len(data_eta_experiment))
    print(len(data_eta_analytical))




    arr = np.full((11, 18), np.nan)
    arr[0:6, 0:2] = data_ct_model
    arr[0:4, 4:6] = data_ct_analytical
    arr[0:10, 2:4] = data_ct_experiment
    arr[0:6, 6:8] = data_cp_model
    arr[0:9, 10:12] = data_cp_analytical
    arr[0:10, 8:10] = data_cp_experiment
    arr[0:6, 12:14] = data_eta_model
    arr[0:6, 16:18] = data_eta_analytical
    arr[0:10, 14:16] = data_eta_experiment
    
    df = pd.DataFrame(arr, columns=columns)
    print(df)

    df.to_csv('form_data_h0_0.75_alpha0_15_deg_phase_90_degree.csv', index=False)

    data_ct_model=data[187:193]
    data_ct_experiment=data[193:201]
    data_ct_analytical=data[201:207]
    print(len(data_ct_model))
    print(len(data_ct_experiment))
    print(len(data_ct_analytical))



    data_cp_model = data[207:213]
    data_cp_experiment = data[213:222]
    data_cp_analytical = data[222:227]

    print(len(data_cp_model))
    print(len(data_cp_experiment))
    print(len(data_cp_analytical))

    data_eta_model=data[227:233]
    data_eta_experiment=data[233:242]
    data_eta_analytical=data[242:]
    print(len(data_eta_model))
    print(len(data_eta_experiment))
    print(len(data_eta_analytical))




    arr = np.full((11, 18), np.nan)
    arr[0:6, 0:2] = data_ct_model
    arr[0:6, 4:6] = data_ct_analytical
    arr[0:8, 2:4] = data_ct_experiment
    arr[0:6, 6:8] = data_cp_model
    arr[0:5, 10:12] = data_cp_analytical
    arr[0:9, 8:10] = data_cp_experiment
    arr[0:6, 12:14] = data_eta_model
    arr[0:6, 16:18] = data_eta_analytical
    arr[0:9, 14:16] = data_eta_experiment
    
    df = pd.DataFrame(arr, columns=columns)
    print(df)

    df.to_csv('form_data_h0_0.75_alpha0_15_deg_phase_75_degree.csv', index=False)




generate_form_data()