import openai
from auto_fingerprint.vector_db import QuadrantClient
from auto_fingerprint.consts import OPEN_AI_MODEL, MAX_TOKENS


class ResponseCollector:
    def __init__(self, vector_db: QuadrantClient, llm: openai.OpenAI):
        self.vector_db = vector_db
        self.llm = llm
        self.responses = {}
        self.chat_history = []

    def tx_version(self):
        queries = [
            "transaction version number definition",
            "transaction version initialization",
            "tx version setting"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        # TODO: add system prompt
        # system_prompt = """
        # You are a code analysis assistant. Your task is to identify the transaction version
        # number used in Bitcoin-related code. Look for:
        # - Version numbers in transaction creation
        # - Default version values
        # - Version constants or definitions
        # Only return a single number, or -1 if the version cannot be determined.
        # """

        user_prompt = "What transaction version number is used in the following code? Only return a number or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS,
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE tx version: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["tx_version"] = int(res) if res != "-1" else -1
        except ValueError:
            self.responses["tx_version"] = -1

    def bip69_sorting(self):
        queries = [
            "BIP69 sorting implementation",
            "transaction input output sorting",
            "lexicographical sorting of transactions",
            "BIP69 compliance check"
        ]
        relevant_chunks = []

        # Gather multiple relevant chunks
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])

        # Deduplicate chunks
        relevant_chunks = list(set(relevant_chunks))

        # TODO: add system prompt
        # system_prompt = """
        # You are a code analysis assistant. Your task is to determine if the code implements
        # BIP69 sorting for Bitcoin transactions. Look for:
        # - Lexicographical sorting of inputs/outputs
        # - References to BIP69 in comments or function names
        # - Sorting of transaction inputs by (txid, vout)
        # - Sorting of outputs by (amount, scriptPubKey)
        # Only return:
        # 1 - if BIP69 sorting is clearly implemented
        # 0 - if BIP69 sorting is clearly not implemented
        # -1 - if it cannot be determined
        # """

        user_prompt = "Does the following code implement BIP69 sorting? Only return 1, 0, or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS,
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE bip69 sorting: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["bip69_sorting"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["bip69_sorting"] = -1

    def mixed_input_types(self):
        queries = [
            "transaction input type mixing",
            "combine different input types",
            "segwit legacy input combination",
            "transaction input validation",
            "input script type checking"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the wallet supports mixing different 
        Bitcoin transaction input types in the same transaction. 
        Look for:
        - Code that handles multiple input types (legacy, segwit, native segwit, taproot)
        - Input type validation or restrictions
        - Transaction building logic that processes different input formats
        - Comments or logic related to input type compatibility
        Only return:
        1 - if mixed input types are clearly supported
        0 - if mixed input types are explicitly prevented
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE mixed input types: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["mixed_input_types"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["mixed_input_types"] = -1

    def input_types(self):
        # Define all possible input types we want to detect
        input_type_queries = [
            "legacy P2PKH input handling",
            "P2SH input implementation",
            "native segwit P2WPKH input",
            "native segwit P2WSH input",
            "P2SH-wrapped segwit input",
            "P2TR taproot input support",
            "multisig input handling",
        ]

        relevant_chunks = []
        for query in input_type_queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to identify which Bitcoin transaction input types are supported.
        Return a list of just comman seperated strings containing only the supported input types from the list above.
        If no input types can be determined, return -1. Do not return any other text.
        Example response: P2PKH, P2WPKH, P2TR
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=100  # Increased token limit for JSON array response
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE input types: ", res)

        self._add_to_chat_history("assistant", res)

        self.responses["input_types"] = res

    def low_r_grinding(self):
        queries = [
            "low R signature grinding",
            "ECDSA signature R value minimization",
            "low R value generation",
            "deterministic ECDSA signature grinding",
            "compact signature generation"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it implements 'Low R' signature grinding for ECDSA signatures.
        Look for:
        - Code that repeatedly generates or modifies signatures to minimize the R value
        - Loops or retries in signature generation aiming for a low R
        - Comments or function names referencing 'low R', 'grinding', or 'compact signatures'
        - Use of deterministic nonce generation with additional grinding logic
        Only return:
        1 - if low R grinding is clearly implemented
        0 - if low R grinding is clearly not implemented
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE low r grinding: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["low_r_grinding"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["low_r_grinding"] = -1

    def change_adress_same_as_input(self):
        queries = [
            "change creation",
            "change address generation",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it allows for change address to be the same as the input scriptpubkey.
        Look for:
        - Code that allows for change address to be the same as the input scriptpubkey
        - Comments or function names referencing 'change', 'change address', or 'change output'
        Only return:
        1 - if change address to be the same as the input scriptpubkey is clearly allowed
        0 - if change address to be the same as the input scriptpubkey is clearly not allowed
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE change address same as input: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["change_address_same_as_input"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["change_address_same_as_input"] = -1

    def address_reuse(self):
        queries = [
            "address reuse",
            "receive address",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it allows for address reuse.
        Look for:
        - Code that allows for address reuse
        - Comments or function names referencing 'address reuse', 'receive address', or 'send address'
        Only return:
        1 - if address reuse is clearly allowed
        0 - if address reuse is clearly not allowed
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE address reuse: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["address_reuse"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["address_reuse"] = -1

    def use_of_nlocktime(self):
        queries = [
            "nlocktime",
            "locktime",
            "transaction locktime",
        ]

        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it allows for the use of nlocktime.
        Look for:
        - Code that allows for the use of nlocktime
        - Comments or function names referencing 'nlocktime', 'locktime', or 'transaction locktime'
        Only return:
        1 - if the use of nlocktime is clearly allowed
        0 - if the use of nlocktime is clearly not allowed
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE use of nlocktime: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["use_of_nlocktime"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["use_of_nlocktime"] = -1

    def nsequence_value(self):
        queries = [
            "nsequence value",
            "sequence number",
            "transaction sequence number",
            "RBF",
            "Replace-by-Fee",
        ]

        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it allows for the use of nsequence.
        Look for:
        - Code that allows for the use of nsequence
        - Comments or function names referencing 'nsequence', 'sequence number', or 'transaction sequence number'
        Only return:
        nsequence value as a hex string, or -1 if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE nsequence value: ", res)

        self._add_to_chat_history("assistant", res)

        self.responses["nsequence_value"] = res

    def change_id_location(self):
        queries = [
            "change output location",
            "change address position",
            "change output index",
            "change output placement",
            "change address generation position"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine where the change output is positioned in Bitcoin transactions.
        Look for:
        - Code that determines the position/index of the change output
        - Logic for placing change output at the end vs. beginning of outputs
        - Comments or function names referencing 'change position', 'change index', or 'change location'
        - Default behavior for change output placement
        Only return:
        0 - if change output is placed at the beginning (index 0)
        1 - if change output is placed at the end (last index)
        2 - if change output position is variable/dynamic
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE change id location: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["change_id_location"] = int(
                res) if res in ["0", "1", "2", "-1"] else -1
        except ValueError:
            self.responses["change_id_location"] = -1

    def op_return_support(self):
        queries = [
            "OP_RETURN output creation",
            "opreturn transaction output",
            "data embedding in transactions",
            "null data output",
            "OP_RETURN script creation",
            "data carrier output"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        # print chunks
        print("Relevant chunks: ", relevant_chunks)

        task_prompt = """
        Analyze the following code to determine if it supports creating OP_RETURN outputs in Bitcoin transactions.
        Look for:
        - Code that creates OP_RETURN outputs or null data outputs
        - Functions that embed data in transactions
        - Script creation for data carrier outputs
        - Comments or function names referencing 'OP_RETURN', 'data output', or 'null data'
        - Transaction building logic that handles data outputs
        Only return:
        1 - if OP_RETURN outputs are clearly supported
        0 - if OP_RETURN outputs are clearly not supported
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE op_return_support: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["op_return_support"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["op_return_support"] = -1

    def output_types(self):
        queries = [
            "output scriptPubKey type creation",
            "destination address type encoding",
            "P2TR taproot output support",
            "P2WPKH output creation",
            "P2SH output script",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to identify which Bitcoin transaction output types are supported.
        Return a comma-separated list of supported output types from: P2PKH, P2SH, P2WPKH, P2WSH, P2TR.
        If no output types can be determined, return -1. Do not return any other text.
        Example response: P2PKH, P2WPKH, P2TR
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=100
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE output types: ", res)

        self._add_to_chat_history("assistant", res)

        self.responses["output_types"] = res

    def number_of_outputs(self):
        queries = [
            "batch payment multiple recipients",
            "changeless transaction no change",
            "transaction output count",
            "single recipient transaction",
            "payment batching implementation",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine the typical number of outputs in transactions.
        Look for:
        - Support for batch payments (multiple recipients in one transaction)
        - Changeless transactions (no change output)
        - Single recipient plus change output
        Return a brief description of the output behavior (e.g., "single recipient + change",
        "supports batching", "changeless transactions supported"). Do not return any other text.
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=100
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE number of outputs: ", res)

        self._add_to_chat_history("assistant", res)

        self.responses["number_of_outputs"] = res

    def compressed_public_keys(self):
        queries = [
            "compressed public key generation",
            "ECDSA public key format",
            "public key serialization compressed",
            "SEC encoded public key",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if it uses compressed public keys.
        Look for:
        - Compressed vs uncompressed public key generation
        - 33-byte (compressed) vs 65-byte (uncompressed) key formats
        - SEC encoding with 0x02/0x03 prefix (compressed) vs 0x04 prefix (uncompressed)
        Only return:
        1 - if compressed public keys are clearly used
        0 - if uncompressed public keys are clearly used
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE compressed public keys: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["compressed_public_keys"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["compressed_public_keys"] = -1

    def input_order_smallest_first(self):
        queries = [
            "coin selection order by amount",
            "UTXO sorting by value ascending",
            "input sorting smallest first",
            "coin selection smallest amount",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if transaction inputs are ordered smallest first by value.
        Look for:
        - UTXO sorting by amount in ascending order
        - Coin selection that prioritizes smaller UTXOs first
        - Input ordering logic based on value
        Only return:
        1 - if inputs are clearly ordered smallest first
        0 - if inputs are clearly not ordered smallest first
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE input order smallest first: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["input_order_smallest_first"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["input_order_smallest_first"] = -1

    def input_order_largest_first(self):
        queries = [
            "coin selection order by amount descending",
            "UTXO sorting by value descending",
            "input sorting largest first",
            "coin selection largest amount",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if transaction inputs are ordered largest first by value.
        Look for:
        - UTXO sorting by amount in descending order
        - Coin selection that prioritizes larger UTXOs first
        - Input ordering logic based on value (largest to smallest)
        Only return:
        1 - if inputs are clearly ordered largest first
        0 - if inputs are clearly not ordered largest first
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE input order largest first: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["input_order_largest_first"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["input_order_largest_first"] = -1

    def input_order_oldest_first(self):
        queries = [
            "UTXO sorting by age oldest first",
            "FIFO oldest first coin selection",
            "coin selection order by confirmation",
            "input sorting by block height",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if transaction inputs are ordered oldest first (FIFO).
        Look for:
        - UTXO sorting by age or confirmation count
        - Coin selection that prioritizes older UTXOs first
        - Input ordering by block height or timestamp
        Only return:
        1 - if inputs are clearly ordered oldest first
        0 - if inputs are clearly not ordered oldest first
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE input order oldest first: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["input_order_oldest_first"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["input_order_oldest_first"] = -1

    def round_fee_indicator(self):
        queries = [
            "manual fee entry user input",
            "custom fee rate setting",
            "fee rate selection interface",
            "round fee calculation",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the wallet uses round fee rates (indicating manual fee entry).
        Look for:
        - User interface for manual fee rate input
        - Round number fee rates (e.g., 1, 5, 10 sat/vB)
        - Custom fee rate settings vs. automatic fee estimation
        Only return:
        1 - if round fee rates or manual fee entry is clearly supported
        0 - if only automatic/non-round fee estimation is used
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE round fee indicator: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["round_fee_indicator"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["round_fee_indicator"] = -1

    def change_type_matches_output(self):
        queries = [
            "change output type selection",
            "change address script type matching",
            "change output matches payment output type",
            "change address type consistency",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the change output type matches the payment output type.
        Look for:
        - Change address type matching the destination/payment output type
        - Logic that ensures change and payment outputs use the same script type
        - Change output type selection based on payment output
        Only return:
        1 - if change type clearly matches the payment output type
        0 - if change type clearly does not match the payment output type
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE change type matches output: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["change_type_matches_output"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["change_type_matches_output"] = -1

    def change_type_matches_input(self):
        queries = [
            "change output type matches input type",
            "change address same script type as input",
            "change output derived from input type",
            "change address type from spending input",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the change output type matches the input type.
        Look for:
        - Change address type matching the input/spending script type
        - Logic that derives change address type from the input UTXOs
        - Change output type selection based on input script type
        Only return:
        1 - if change type clearly matches the input type
        0 - if change type clearly does not match the input type
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE change type matches input: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["change_type_matches_input"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["change_type_matches_input"] = -1

    def spend_unconfirmed(self):
        queries = [
            "spend unconfirmed change output",
            "minimum confirmations for spending",
            "unconfirmed UTXO spending policy",
            "zero confirmation transaction input",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the wallet allows spending unconfirmed outputs.
        Look for:
        - Code that allows or prevents spending unconfirmed/zero-confirmation UTXOs
        - Minimum confirmation requirements for spending
        - Unconfirmed change output spending policy
        Only return:
        1 - if spending unconfirmed outputs is clearly allowed
        0 - if spending unconfirmed outputs is clearly not allowed
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE spend unconfirmed: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["spend_unconfirmed"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["spend_unconfirmed"] = -1

    def rbf_replacement(self):
        queries = [
            "RBF replacement transaction creation",
            "fee bumping transaction replacement",
            "replace by fee bump implementation",
            "transaction replacement broadcast",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the wallet supports creating RBF replacement transactions
        (i.e., actually bumping fees by creating and broadcasting a replacement transaction).
        This is distinct from simply signaling RBF via nSequence.
        Look for:
        - Code that creates replacement transactions with higher fees
        - Fee bumping functionality
        - Transaction replacement broadcasting
        Only return:
        1 - if RBF replacement transaction creation is clearly supported
        0 - if RBF replacement transaction creation is clearly not supported
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE rbf replacement: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["rbf_replacement"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["rbf_replacement"] = -1

    def feerate_estimation_source(self):
        queries = [
            "fee estimation source API",
            "estimatesmartfee fee rate",
            "fee rate provider external",
            "mempool fee estimation",
            "block target fee calculation",
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine the source of fee rate estimation.
        Look for:
        - External API calls for fee estimation (e.g., mempool.space, blockstream, bitcoinfees)
        - Bitcoin Core's estimatesmartfee RPC usage
        - Built-in fee estimation algorithms
        - Fee rate provider configuration
        Return a brief description of the fee estimation source (e.g., "Bitcoin Core estimatesmartfee",
        "mempool.space API", "built-in estimation"). If it cannot be determined, return -1.
        Do not return any other text.
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=100
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE feerate estimation source: ", res)

        self._add_to_chat_history("assistant", res)

        self.responses["feerate_estimation_source"] = res

    def _add_to_chat_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})
