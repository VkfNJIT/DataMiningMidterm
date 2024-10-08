import os
import sys
from itertools import combinations # important for getting all the possible k itemsets
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

def item_k_support_possibilities(item_names, k):

    item_k_arrange = combinations(item_names, k)
    possibilities_of_k_items = [item for item in item_k_arrange]
    return possibilities_of_k_items

def count_itemsets_for_k(current_itemset, transactions, k):
    item_k_filter = [name for name in current_itemset.keys()]
    item_k_frequent_names = item_k_support_possibilities(item_k_filter, k)
    itemset_k = {}
    for item in item_k_frequent_names:
        count_occ = sum(1 for transact in transactions if set(item).issubset(transact))
        itemset_k[tuple(item)] = float(count_occ) / len(transactions)
    return itemset_k
    
def get_itemsets_with_confidence(total_itemset_frequent, min_confidence):
    itemset_confidence = {}
    itemset_copy = total_itemset_frequent.copy()
    for key, val in total_itemset_frequent.items():
        if isinstance(key, tuple):
            if len(key) == 2:
                first = key[0]
                second = key[-1]
                confidence_val = val / total_itemset_frequent[first]
                if confidence_val >= min_confidence:
                    itemset_confidence[(first, second)] = confidence_val
                first_reverse = key[-1]
                second_reverse = key[0]
                confidence_val = val / itemset_copy[first_reverse]
                if confidence_val >= min_confidence:
                    itemset_confidence[(first_reverse, second_reverse)] = confidence_val
                    itemset_copy[(first_reverse, second_reverse)] = val
                
            elif len(key) > 2:    
                for i in range(1, len(key)+1):
                    for first in combinations(list(key), i):
                        second = tuple(set(key).difference(set(first))) # This will get what comes after ->
                        if len(second) > 0:
                            first = tuple(sorted(first))
                            if first in itemset_copy:
                                
                                confidence_val = float(val)/itemset_copy[first]
                                if confidence_val >= min_confidence:
                                    itemset_confidence[(first, second)] = confidence_val
                                    itemset_copy[(first, second)] = val
                            else:
                                if len(first) == 1:
                                    confidence_val = val / itemset_copy[first[0]]
                                    if confidence_val >= min_confidence:
                                        itemset_confidence[(first, second)] = confidence_val
                                        itemset_copy[(first, second)] = val
                                else:
                                    for item in total_itemset_frequent.keys():
                                        if len(set(item).difference(set(first))) == 0:
                                            confidence_val = val / itemset_copy[item]
                                            if confidence_val >= min_confidence:
                                                itemset_confidence[(first, second)] = confidence_val
                                                itemset_copy[(first, second)] = val         
         
    return itemset_confidence, itemset_copy

def collect_frequent_itemset(unfilter_dict_k, min_support):
    filtered_dict = {}
    for key, val in unfilter_dict_k.items():
        if val >= min_support:
            itemsetkey = key
            filtered_dict[key] = val
    return filtered_dict

      



selected_stores = {1: "amazon", 2: "best_buy", 3: "k-mart", 4: "nike", 5: "ace_hardware"}
try:
    selected_id = int(input(
    "Enter the store number for the dataset that you want:\n1. Amazon\n2. Best Buy\n3. K-mart\n4. Nike\n5. Ace Hardware\n"))
    if selected_id not in selected_stores.keys():
        print("invalid number, There are only 5 choices!Try again next time")
        sys.exit()
except ValueError:
    print("Invalid input! There are only 5 choices, please enter a valid number(1 to 5) next time")
    sys.exit()
item_names = pd.read_csv(f"{os.getcwd()}/Itemsets/{selected_stores[selected_id]}_items.csv")
transactions = pd.read_csv(f"{os.getcwd()}/Itemsets/{selected_stores[selected_id]}_transactions.csv")
print(f"You have selected the {selected_stores[selected_id]} dataset")

# Enter the minimum support and the minimum confidence 
min_support = float(input("Please enter the minimum support percent that you want (1 to 100):\n"))
min_support /= 100
min_confidence = float(input("Please enter the minimum confidence percent that you want (1 to 100):\n"))
min_confidence /= 100

itemset_k1 = item_names.set_index("Item Name").to_dict()["Item #"]

# This technique Only for the itemsets where k = 1 
# Split the string by comma to seperate each string in a row
item_k1_names = [name for name in item_names["Item Name"]]

item_k1_count = transactions['Transaction'].str.split(", ").explode().value_counts()

item_k1 = item_k1_count.to_dict()

# Get the support value for each itemset-1
for k, _ in itemset_k1.items():
    if k not in item_k1:
        itemset_k1[k] = float(0)
    else: 
        itemset_k1[k] = float(item_k1[k]) / len(transactions["Transaction"])
itemset_frequent_k1 = collect_frequent_itemset(itemset_k1, min_support)
item_k = transactions['Transaction'].str.split(", ").to_list()
itemset_k = {}
itemset_frequent_k = itemset_frequent_k1
k_val = 2
updated_itemset = itemset_frequent_k1
while len(itemset_frequent_k) >= k_val:
    itemset_k = count_itemsets_for_k(itemset_frequent_k1, item_k, k_val)
    itemset_frequent_k = collect_frequent_itemset(itemset_k, min_support)
    updated_itemset.update(itemset_frequent_k)
    k_val += 1
print()
for key_s, val_s in updated_itemset.items():
    print(f"Itemset: {key_s}, Support: {val_s}\n")
item_conf, item_supp = get_itemsets_with_confidence(updated_itemset, min_confidence)
print()
rule_ci = 1
for key_c,val_c in item_conf.items():
    if len(key_c) == 2 and val_c > 0:
        print(
        f"Rule {rule_ci}:{set(key_c[0:1])} -> {set(key_c[1:])}\nConfidence: {val_c*100:.2f}%\nSupport: {item_supp[key_c]*100:.2f}%")
        rule_ci += 1
        print()
    else:
        for i in range(len(key_c)-1):
            if len(val_c) > 0:
                print(
        f"Rule {rule_ci}:{set(key_c[0:i+1])} -> {set(key_c[i+1:])}\nConfidence: {val_c*100:.2f}%\nSupport: {item_supp[key_c]*100:.2f}%")
                rule_ci += 1
                print()




te = TransactionEncoder()
te_ary = te.fit(item_k).transform(item_k)
dataframe = pd.DataFrame(te_ary, columns=te.columns_)
checking_apriori = apriori(dataframe, min_support=min_support, use_colnames=True)
print()
print("Apriori Library")
print(checking_apriori)
print()
checking_fpgrowth = fpgrowth(dataframe, min_support=min_support, use_colnames=True)
print("FP Tree Library")
print(checking_fpgrowth)
if len(checking_apriori.index) > 0:
    ar = association_rules(checking_apriori, metric='confidence', min_threshold=min_confidence)
    print()
    print("Association Rules Library")
    ar = ar[['antecedents', 'consequents', 'support', 'confidence']]
    print(ar)
