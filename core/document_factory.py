import os
import json
from loguru import logger

class DocumentFactor:
    def __init__(self, source):
        if os.path.isdir(source):
            self.source = source
        else:
            raise NotImplementedError(
                "Implementation not provided for sources different from directory"
            )

    def get_document(self, docid):
        document_path = os.path.join(self.source, f"{docid}.json")
        message = f"The document {docid} doesn't exist"
        if not os.path.isfile(document_path):
            logger.error(message)
            raise ValueError(message)
        contents = json.loads(open(document_path).read())["contents"]

        return contents
