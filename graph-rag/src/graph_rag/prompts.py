"""
Heuristic definitions for Bitcoin wallet fingerprinting.

Prompts ported from scanner/src/auto_fingerprint/response.py to keep both
pipelines consistent. Each heuristic entry has:
  key          - result dict key
  queries      - list of semantic search strings
  prompt       - LLM question
  type         - "binary" (returns int) or "text" (returns str)
  valid_values - tuple of accepted str values for binary; None means any int
  max_tokens   - optional override (text heuristics default to 100)
"""

SYSTEM_PROMPT = """You are a Bitcoin wallet fingerprinting analyst. You analyze wallet source code \
to determine transaction construction behaviors that can identify which wallet software created a transaction.

These fingerprints include: transaction version, input/output types, BIP69 sorting, low-R signature \
grinding, nLockTime anti-fee-sniping, nSequence values, change output positioning, RBF signaling, \
fee estimation sources, and more.

You will be given extracted source code functions from a wallet and asked specific questions about \
its behavior. Base your answer ONLY on the code provided. If the code does not contain enough \
evidence, return -1.

When answering, think step by step about what the code does, then provide ONLY the final answer \
on the last line with no other text. For example, if asked for 1/0/-1, your entire response \
should be just the number."""

FEW_SHOT_EXAMPLES = [
    {
        "user": """Does the following code implement low-R signature grinding? Only return 1, 0, or -1 if unclear.

bool CKey::Sign(const uint256 &hash, std::vector<unsigned char>& vchSig, bool grind, uint32_t test_case) const {
    ...
    unsigned char extra_entropy[32] = {0};
    WriteLE32(extra_entropy, test_case);
    secp256k1_ecdsa_sign(secp256k1_context_sign, &sig, hash.begin(), begin(), secp256k1_nonce_function_rfc6979, grind ? extra_entropy : nullptr);
    ...
    // Grind for low R
    while (IsLowR(vchSig) == false && grind) {
        test_case++;
        ...
    }
}""",
        "assistant": "1",
    },
    {
        "user": """Does the following code implement BIP69 sorting? Only return 1, 0, or -1 if unclear.

void CWallet::AvailableCoins(std::vector<COutput>& vCoins, ...) const {
    ...
    for (const auto& entry : mapWallet) {
        ...
        vCoins.push_back(COutput(pcoin, i, nDepth, ...));
    }
}

// Shuffle outputs
std::shuffle(vCoins.begin(), vCoins.end(), FastRandomContext());""",
        "assistant": "0",
    },
    {
        "user": """Analyze the following code to identify which Bitcoin transaction input types are supported.
Return a list of just comma separated strings containing only the supported input types.
If no input types can be determined, return -1. Do not return any other text.

def serialize_input(self, txin):
    if txin.script_type == 'p2pkh':
        return self._serialize_p2pkh(txin)
    elif txin.script_type in ('p2wpkh', 'p2wsh'):
        return self._serialize_witness(txin)
    elif txin.script_type == 'p2sh':
        return self._serialize_p2sh(txin)
    elif txin.script_type == 'p2wpkh-p2sh':
        return self._serialize_p2sh_p2wpkh(txin)""",
        "assistant": "P2PKH, P2SH, P2WPKH, P2WSH, P2SH-P2WPKH",
    },
]

HEURISTICS: list[dict] = [
    {
        "key": "tx_version",
        "queries": [
            "transaction version number definition",
            "transaction version initialization",
            "tx version setting",
        ],
        "prompt": "What transaction version number is used in the following code? Only return a number or -1 if unclear.",
        "type": "binary",
        "valid_values": None,  # any parseable int is valid
    },
    {
        "key": "bip69_sorting",
        "queries": [
            "BIP69 sorting implementation",
            "transaction input output sorting",
            "lexicographical sorting of transactions",
            "BIP69 compliance check",
        ],
        "prompt": "Does the following code implement BIP69 sorting? Only return 1, 0, or -1 if unclear.",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "mixed_input_types",
        "queries": [
            "transaction input type mixing",
            "combine different input types",
            "segwit legacy input combination",
            "transaction input validation",
            "input script type checking",
        ],
        "prompt": """Analyze the following code to determine if the wallet supports mixing different
Bitcoin transaction input types in the same transaction.
Look for:
- Code that handles multiple input types (legacy, segwit, native segwit, taproot)
- Input type validation or restrictions
- Transaction building logic that processes different input formats
- Comments or logic related to input type compatibility
Only return:
1 - if mixed input types are clearly supported
0 - if mixed input types are explicitly prevented
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "input_types",
        "queries": [
            "legacy P2PKH input handling",
            "P2SH input implementation",
            "native segwit P2WPKH input",
            "native segwit P2WSH input",
            "P2SH-wrapped segwit input",
            "P2TR taproot input support",
            "multisig input handling",
        ],
        "prompt": """Analyze the following code to identify which Bitcoin transaction input types are supported.
Return a list of just comma separated strings containing only the supported input types from the list above.
If no input types can be determined, return -1. Do not return any other text.
Example response: P2PKH, P2WPKH, P2TR""",
        "type": "text",
        "max_tokens": 100,
    },
    {
        "key": "low_r_grinding",
        "queries": [
            "low R signature grinding",
            "ECDSA signature R value minimization",
            "low R value generation",
            "deterministic ECDSA signature grinding",
            "compact signature generation",
        ],
        "prompt": """Analyze the following code to determine if it implements 'Low R' signature grinding for ECDSA signatures.
Look for:
- Code that repeatedly generates or modifies signatures to minimize the R value
- Loops or retries in signature generation aiming for a low R
- Comments or function names referencing 'low R', 'grinding', or 'compact signatures'
- Use of deterministic nonce generation with additional grinding logic
Only return:
1 - if low R grinding is clearly implemented
0 - if low R grinding is clearly not implemented
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "change_address_same_as_input",
        "queries": [
            "change creation",
            "change address generation",
        ],
        "prompt": """Analyze the following code to determine if it allows for change address to be the same as the input scriptpubkey.
Look for:
- Code that allows for change address to be the same as the input scriptpubkey
- Comments or function names referencing 'change', 'change address', or 'change output'
Only return:
1 - if change address to be the same as the input scriptpubkey is clearly allowed
0 - if change address to be the same as the input scriptpubkey is clearly not allowed
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "address_reuse",
        "queries": [
            "address reuse",
            "receive address",
        ],
        "prompt": """Analyze the following code to determine if it allows for address reuse.
Look for:
- Code that allows for address reuse
- Comments or function names referencing 'address reuse', 'receive address', or 'send address'
Only return:
1 - if address reuse is clearly allowed
0 - if address reuse is clearly not allowed
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "use_of_nlocktime",
        "queries": [
            "nlocktime",
            "locktime",
            "transaction locktime",
        ],
        "prompt": """Analyze the following code to determine if it allows for the use of nlocktime.
Look for:
- Code that allows for the use of nlocktime
- Comments or function names referencing 'nlocktime', 'locktime', or 'transaction locktime'
Only return:
1 - if the use of nlocktime is clearly allowed
0 - if the use of nlocktime is clearly not allowed
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "nsequence_value",
        "queries": [
            "nsequence value",
            "sequence number",
            "transaction sequence number",
            "RBF",
            "Replace-by-Fee",
        ],
        "prompt": """Analyze the following code to determine if it allows for the use of nsequence.
Look for:
- Code that allows for the use of nsequence
- Comments or function names referencing 'nsequence', 'sequence number', or 'transaction sequence number'
Only return:
nsequence value as a hex string, or -1 if it cannot be determined""",
        "type": "text",
        "max_tokens": 16,
    },
    {
        "key": "change_id_location",
        "queries": [
            "change output location",
            "change address position",
            "change output index",
            "change output placement",
            "change address generation position",
        ],
        "prompt": """Analyze the following code to determine where the change output is positioned in Bitcoin transactions.
Look for:
- Code that determines the position/index of the change output
- Logic for placing change output at the end vs. beginning of outputs
- Comments or function names referencing 'change position', 'change index', or 'change location'
- Default behavior for change output placement
Only return:
0 - if change output is placed at the beginning (index 0)
1 - if change output is placed at the end (last index)
2 - if change output position is variable/dynamic
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "2", "-1"),
    },
    {
        "key": "op_return_support",
        "queries": [
            "OP_RETURN output creation",
            "opreturn transaction output",
            "data embedding in transactions",
            "null data output",
            "OP_RETURN script creation",
            "data carrier output",
        ],
        "prompt": """Analyze the following code to determine if it supports creating OP_RETURN outputs in Bitcoin transactions.
Look for:
- Code that creates OP_RETURN outputs or null data outputs
- Functions that embed data in transactions
- Script creation for data carrier outputs
- Comments or function names referencing 'OP_RETURN', 'data output', or 'null data'
- Transaction building logic that handles data outputs
Only return:
1 - if OP_RETURN outputs are clearly supported
0 - if OP_RETURN outputs are clearly not supported
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "output_types",
        "queries": [
            "output scriptPubKey type creation",
            "destination address type encoding",
            "P2TR taproot output support",
            "P2WPKH output creation",
            "P2SH output script",
        ],
        "prompt": """Analyze the following code to identify which Bitcoin transaction output types are supported.
Return a comma-separated list of supported output types from: P2PKH, P2SH, P2WPKH, P2WSH, P2TR.
If no output types can be determined, return -1. Do not return any other text.
Example response: P2PKH, P2WPKH, P2TR""",
        "type": "text",
        "max_tokens": 100,
    },
    {
        "key": "number_of_outputs",
        "queries": [
            "batch payment multiple recipients",
            "changeless transaction no change",
            "transaction output count",
            "single recipient transaction",
            "payment batching implementation",
        ],
        "prompt": """Analyze the following code to determine the typical number of outputs in transactions.
Look for:
- Support for batch payments (multiple recipients in one transaction)
- Changeless transactions (no change output)
- Single recipient plus change output
Return a brief description of the output behavior (e.g., "single recipient + change",
"supports batching", "changeless transactions supported"). Do not return any other text.""",
        "type": "text",
        "max_tokens": 100,
    },
    {
        "key": "compressed_public_keys",
        "queries": [
            "compressed public key generation",
            "ECDSA public key format",
            "public key serialization compressed",
            "SEC encoded public key",
        ],
        "prompt": """Analyze the following code to determine if it uses compressed public keys.
Look for:
- Compressed vs uncompressed public key generation
- 33-byte (compressed) vs 65-byte (uncompressed) key formats
- SEC encoding with 0x02/0x03 prefix (compressed) vs 0x04 prefix (uncompressed)
Only return:
1 - if compressed public keys are clearly used
0 - if uncompressed public keys are clearly used
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "input_order_smallest_first",
        "queries": [
            "coin selection order by amount",
            "UTXO sorting by value ascending",
            "input sorting smallest first",
            "coin selection smallest amount",
        ],
        "prompt": """Analyze the following code to determine if transaction inputs are ordered smallest first by value.
Look for:
- UTXO sorting by amount in ascending order
- Coin selection that prioritizes smaller UTXOs first
- Input ordering logic based on value
Only return:
1 - if inputs are clearly ordered smallest first
0 - if inputs are clearly not ordered smallest first
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "input_order_largest_first",
        "queries": [
            "coin selection order by amount descending",
            "UTXO sorting by value descending",
            "input sorting largest first",
            "coin selection largest amount",
        ],
        "prompt": """Analyze the following code to determine if transaction inputs are ordered largest first by value.
Look for:
- UTXO sorting by amount in descending order
- Coin selection that prioritizes larger UTXOs first
- Input ordering logic based on value (largest to smallest)
Only return:
1 - if inputs are clearly ordered largest first
0 - if inputs are clearly not ordered largest first
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "input_order_oldest_first",
        "queries": [
            "UTXO sorting by age oldest first",
            "FIFO oldest first coin selection",
            "coin selection order by confirmation",
            "input sorting by block height",
        ],
        "prompt": """Analyze the following code to determine if transaction inputs are ordered oldest first (FIFO).
Look for:
- UTXO sorting by age or confirmation count
- Coin selection that prioritizes older UTXOs first
- Input ordering by block height or timestamp
Only return:
1 - if inputs are clearly ordered oldest first
0 - if inputs are clearly not ordered oldest first
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "round_fee_indicator",
        "queries": [
            "manual fee entry user input",
            "custom fee rate setting",
            "fee rate selection interface",
            "round fee calculation",
        ],
        "prompt": """Analyze the following code to determine if the wallet uses round fee rates (indicating manual fee entry).
Look for:
- User interface for manual fee rate input
- Round number fee rates (e.g., 1, 5, 10 sat/vB)
- Custom fee rate settings vs. automatic fee estimation
Only return:
1 - if round fee rates or manual fee entry is clearly supported
0 - if only automatic/non-round fee estimation is used
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "change_type_matches_output",
        "queries": [
            "change output type selection",
            "change address script type matching",
            "change output matches payment output type",
            "change address type consistency",
        ],
        "prompt": """Analyze the following code to determine if the change output type matches the payment output type.
Look for:
- Change address type matching the destination/payment output type
- Logic that ensures change and payment outputs use the same script type
- Change output type selection based on payment output
Only return:
1 - if change type clearly matches the payment output type
0 - if change type clearly does not match the payment output type
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "change_type_matches_input",
        "queries": [
            "change output type matches input type",
            "change address same script type as input",
            "change output derived from input type",
            "change address type from spending input",
        ],
        "prompt": """Analyze the following code to determine if the change output type matches the input type.
Look for:
- Change address type matching the input/spending script type
- Logic that derives change address type from the input UTXOs
- Change output type selection based on input script type
Only return:
1 - if change type clearly matches the input type
0 - if change type clearly does not match the input type
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "spend_unconfirmed",
        "queries": [
            "spend unconfirmed change output",
            "minimum confirmations for spending",
            "unconfirmed UTXO spending policy",
            "zero confirmation transaction input",
        ],
        "prompt": """Analyze the following code to determine if the wallet allows spending unconfirmed outputs.
Look for:
- Code that allows or prevents spending unconfirmed/zero-confirmation UTXOs
- Minimum confirmation requirements for spending
- Unconfirmed change output spending policy
Only return:
1 - if spending unconfirmed outputs is clearly allowed
0 - if spending unconfirmed outputs is clearly not allowed
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "rbf_replacement",
        "queries": [
            "RBF replacement transaction creation",
            "fee bumping transaction replacement",
            "replace by fee bump implementation",
            "transaction replacement broadcast",
        ],
        "prompt": """Analyze the following code to determine if the wallet supports creating RBF replacement transactions
(i.e., actually bumping fees by creating and broadcasting a replacement transaction).
This is distinct from simply signaling RBF via nSequence.
Look for:
- Code that creates replacement transactions with higher fees
- Fee bumping functionality
- Transaction replacement broadcasting
Only return:
1 - if RBF replacement transaction creation is clearly supported
0 - if RBF replacement transaction creation is clearly not supported
-1 - if it cannot be determined""",
        "type": "binary",
        "valid_values": ("0", "1", "-1"),
    },
    {
        "key": "feerate_estimation_source",
        "queries": [
            "fee estimation source API",
            "estimatesmartfee fee rate",
            "fee rate provider external",
            "mempool fee estimation",
            "block target fee calculation",
        ],
        "prompt": """Analyze the following code to determine the source of fee rate estimation.
Look for:
- External API calls for fee estimation (e.g., mempool.space, blockstream, bitcoinfees)
- Bitcoin Core's estimatesmartfee RPC usage
- Built-in fee estimation algorithms
- Fee rate provider configuration
Return a brief description of the fee estimation source (e.g., "Bitcoin Core estimatesmartfee",
"mempool.space API", "built-in estimation"). If it cannot be determined, return -1.
Do not return any other text.""",
        "type": "text",
        "max_tokens": 100,
    },
]
