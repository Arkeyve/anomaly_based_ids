import pandas as pd
import numpy as np

kdd = []
kdd_labels = [
    "duration",                     # int
    "protocol_type",                # str
    "service",                      # str
    "flag",                         # str
    "src_bytes",                    # int
    "dst_bytes",                    # int
    "land",                         # int
    "wrong_fragment",               # int
    "urgent",                       # int
    "hot",                          # int
    "num_failed_logins",            # int
    "logged_in",                    # int
    "num_compromised",              # int
    "root_shell",                   # int
    "su_attempted",                 # int
    "num_root",                     # int
    "num_file_creations",           # int
    "num_shells",                   # int
    "num_access_files",             # int
    "num_outbound_cmds",            # int
    "is_host_login",                # int
    "is_guest_login",               # int
    "count",                        # int
    "srv_count",                    # int
    "serror_rate",                  # float
    "srv_serror_rate",              # float
    "rerror_rate",                  # float
    "srv_rerror_rate",              # float
    "same_srv_rate",                # float
    "diff_srv_rate",                # float
    "srv_diff_host_rate",           # float
    "dst_host_count",               # int
    "dst_host_srv_count",           # int
    "dst_host_same_srv_rate",       # float
    "dst_host_diff_srv_rate",       # float
    "dst_host_same_src_port_rate",  # float
    "dst_host_srv_diff_host_rate",  # float
    "dst_host_serror_rate",         # float
    "dst_host_srv_serror_rate",     # float
    "dst_host_rerror_rate",         # float
    "dst_host_srv_rerror_rate",     # float
    "attack_type"                   # str
]

print("importing KDD'99 database...")
kdd = pd.read_csv('../dataset/kddcup.data_10_percent', sep = ',', header = None)
kdd.columns = kdd_labels
curr_rows = kdd.shape[0]
print("KDD'99 rows:", curr_rows)

print("\nremoving duplicates...")
kdd = kdd.drop_duplicates(keep='first')
later_rows = kdd.shape[0]
print("KDD'99 rows:", later_rows)
print("reduction of", curr_rows - later_rows, "rows")

print("\nreplacing string values with numeric...")
replace_dictionary = dict.fromkeys(['protocol_type', 'service', 'flag', 'attack_type'])
for attribute in replace_dictionary.keys():
    attr_values = kdd.loc[:, attribute].drop_duplicates()
    replace_dictionary[attribute] = dict( zip( attr_values, np.arange(0.001, (len(attr_values) + 1) * 0.001, 0.001) ) )
print(replace_dictionary)
kdd = kdd.replace(replace_dictionary)
