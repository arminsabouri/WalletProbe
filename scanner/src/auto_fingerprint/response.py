
import json
import lmdb
import openai
from auto_fingerprint.vector_db import QuadrantClient
from auto_fingerprint.consts import OPEN_AI_MODEL, MAX_TOKENS


class ResponseCollector:
    def __init__(self, vector_db: QuadrantClient, llm: openai.OpenAI):
        self.vector_db = vector_db
        self.llm = llm
        self.responses = {}
        self.chat_history = []

    def save_results(self, lmdb_env: lmdb.Environment, wallet_name: str, wallet_version: str):
        key = f"{wallet_name}:{wallet_version}"
        value = json.dumps(self.responses)
        with lmdb_env.begin(write=True) as txn:
            txn.put(key.encode('utf-8'), value.encode('utf-8'))

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

        system_prompt = """
        You are a code analysis assistant. Your task is to identify the transaction version 
        number used in Bitcoin-related code. Look for:
        - Version numbers in transaction creation
        - Default version values
        - Version constants or definitions
        Only return a single number, or -1 if the version cannot be determined.
        """

        user_prompt = "What transaction version number is used in the following code? Only return a number or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
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

        system_prompt = """
        You are a code analysis assistant. Your task is to determine if the code implements 
        BIP69 sorting for Bitcoin transactions. Look for:
        - Lexicographical sorting of inputs/outputs
        - References to BIP69 in comments or function names
        - Sorting of transaction inputs by (txid, vout)
        - Sorting of outputs by (amount, scriptPubKey)
        Only return:
        1 - if BIP69 sorting is clearly implemented
        0 - if BIP69 sorting is clearly not implemented
        -1 - if it cannot be determined
        """

        user_prompt = "Does the following code implement BIP69 sorting? Only return 1, 0, or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=MAX_TOKENS
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

    def _add_to_chat_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

