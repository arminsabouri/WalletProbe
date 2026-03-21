HEURISTICS = {
    "independent": [
        {"key": "tx_version", "label": "Transaction Version"},
        {"key": "input_types", "label": "Input Types"},
        {"key": "mixed_input_types", "label": "Mixed Input Types"},
        {"key": "output_types", "label": "Output Types"},
        {"key": "number_of_outputs", "label": "Number of Outputs"},
        {"key": "nsequence_value", "label": "nSequence Value (RBF Signaling)"},
        {"key": "compressed_public_keys", "label": "Compressed Public Keys"},
        {"key": "use_of_nlocktime", "label": "Anti-Fee-Sniping (nLockTime)"},
        {"key": "op_return_support", "label": "OP_RETURN Support"},
        {"key": "address_reuse", "label": "Address Reuse"},
        {"key": "low_r_grinding", "label": "Low-R Grinding"},
    ],
    "probabilistic": [
        {"key": "bip69_sorting", "label": "BIP 69 Sorting"},
        {"key": "input_order_smallest_first", "label": "Input Order: Smallest First"},
        {"key": "input_order_largest_first", "label": "Input Order: Largest First"},
        {"key": "input_order_oldest_first", "label": "Input Order: Oldest First"},
        {"key": "round_fee_indicator", "label": "Round Fee Indicator"},
    ],
    "dependent": [
        {"key": "change_id_location", "label": "Change Position in Outputs"},
        {"key": "change_address_same_as_input", "label": "Change Address Same as Input"},
        {"key": "change_type_matches_output", "label": "Change Type Matches Output"},
        {"key": "change_type_matches_input", "label": "Change Type Matches Input"},
    ],
    "temporal": [
        {"key": "spend_unconfirmed", "label": "Spend Unconfirmed"},
        {"key": "rbf_replacement", "label": "RBF Replacement"},
        {"key": "feerate_estimation_source", "label": "Feerate Estimation Source"},
    ],
}
