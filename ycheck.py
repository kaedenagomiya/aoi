import numpy as np
import toybox

eval_json_path = 'result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/eval4mid_LJ_V1.json'
a = toybox.load_json(eval_json_path)

dt_list = [a[n]['dt'] for n in range(len(a))]
RTF4mel_list = [a[n]['RTF4mel'] for n in range(len(a))]
utmos_list = [a[n]['utmos'] for n in range(len(a))]

dt_nparr = np.array(dt_list[1:101])
RTF4mel_nparr = np.array(RTF4mel_list[1:101])
utmos_nparr = np.array(utmos_list[1:101])
print(len(dt_nparr))

significant_digits = 8

# for culc difference time to infer text2mel
dt_mean = toybox.round_significant_digits(np.mean(dt_nparr), significant_digits=significant_digits)
dt_var = toybox.round_significant_digits(np.var(dt_nparr), significant_digits=significant_digits)
dt_std = toybox.round_significant_digits(np.std(dt_nparr), significant_digits=significant_digits)
print(f'dt ---------------------------')
print(f'dt mean: {dt_mean}')
print(f'dt var: {dt_var}')
print(f'dt std: {dt_std}')

# for culc RTF4mel to infer text2mel
RTF4mel_mean = toybox.round_significant_digits(np.mean(RTF4mel_nparr), significant_digits=significant_digits)
RTF4mel_var = toybox.round_significant_digits(np.var(RTF4mel_nparr), significant_digits=significant_digits)
RTF4mel_std = toybox.round_significant_digits(np.std(RTF4mel_nparr), significant_digits=significant_digits)
print(f'RTF ---------------------------')
print(f'RTF mean: {RTF4mel_mean}')
print(f'RTF var: {RTF4mel_var}')
print(f'RTF std: {RTF4mel_std}')

# for culc utmos to infer
print(f'utmos ---------------------------')
utmos_mean = toybox.round_significant_digits(np.mean(utmos_nparr), significant_digits=significant_digits)
utmos_var = toybox.round_significant_digits(np.var(utmos_nparr), significant_digits=significant_digits)
utmos_std = toybox.round_significant_digits(np.std(utmos_nparr), significant_digits=significant_digits)
print(f'utmos mean: {utmos_mean}')
print(f'utmos var: {utmos_var}')
print(f'utmos std: {utmos_std}')
