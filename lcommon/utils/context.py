import numpy as np

from lcommon.protocols import Dataset
from lcommon.types import AnnotatedConversation, Context

_DEFAULT_ROLE = "user"


def parse_context(context: Context) -> list[AnnotatedConversation]:
    """
    Parses the context input and returns it as a list of `AnnotatedConversation` objects.

    ### Parameters
    ----------
    `context`: the context to parse.

    ### Returns
    -------
    The parsed context as a list of `AnnotatedConversation` objects.
    - If the length of the outer list is 1, then the context is a single message/conversation.

    ### Raises
    -------
    `ValueError`: if the input type is not supported.
    `ValueError`: if a message dictionary does not contain a `content` field.
    `AssertionError`: if a context list is provided and it is empty.
    """

    if isinstance(context, list):
        assert len(context) > 0, "the context list must not be empty."

    # Single message
    if isinstance(context, str):
        return [[{"content": context, "role": _DEFAULT_ROLE}]]
    elif isinstance(context, dict):  # with model-specific fields
        if "content" not in context:
            raise ValueError("The message dictionary must contain a `content` field.")
        if "role" not in context:
            context["role"] = _DEFAULT_ROLE
        return [[context]]

    # Single conversation
    elif (
        (isinstance(context, np.ndarray) and context.ndim == 1)
        or isinstance(context, list)
    ) and all(isinstance(context[i], str) for i in range(len(context))):
        return [[{"content": input, "role": _DEFAULT_ROLE} for input in context]]  # type: ignore[reportReturnType]
    elif (
        (isinstance(context, np.ndarray) and context.ndim == 1)
        or isinstance(context, list)
    ) and all(
        isinstance(context[i], dict) for i in range(len(context))
    ):  # with model-specific fields
        for message in context:
            if "content" not in message:
                raise ValueError(
                    "All message dictionaries must contain a `content` field."
                )
            if "role" not in message:
                message["role"] = _DEFAULT_ROLE  # type: ignore
        return [context]  # type: ignore[reportReturnType]

    # Multiple messages/conversations
    elif (
        (
            (isinstance(context, np.ndarray) and context.ndim == 2)
            or isinstance(context, list)
        )
        and all(
            (
                (isinstance(context[i], np.ndarray) and context[i].ndim == 1)  # type: ignore[reportAttributeAccessIssue]
                or isinstance(context[i], list)
            )
            for i in range(len(context))
        )
        and all(
            isinstance(context[i][j], str)  # type: ignore[reportArgumentType]
            for i, j in np.ndindex((len(context), len(context[0])))
        )
    ):
        return [  # type: ignore[reportReturnType]
            [{"content": message, "role": _DEFAULT_ROLE} for message in conversation]
            for conversation in context
        ]
    elif (
        (
            (isinstance(context, np.ndarray) and context.ndim == 2)
            or isinstance(context, list)
        )
        and all(
            (
                (isinstance(context[i], np.ndarray) and context[i].ndim == 1)  # type: ignore[reportAttributeAccessIssue]
                or isinstance(context[i], list)
            )
            for i in range(len(context))
        )
        and all(
            isinstance(context[i][j], dict)  # type: ignore[reportArgumentType]
            for i, j in np.ndindex((len(context), len(context[0])))
        )
    ):
        for conversation in context:
            for message in conversation:
                if "content" not in message:
                    raise ValueError(
                        "All message dictionaries must contain a `content` field."
                    )
                if "role" not in message:
                    message["role"] = _DEFAULT_ROLE  # type: ignore[reportIndexIssue]
        return context  # type: ignore[reportReturnType]
    elif isinstance(context, Dataset):
        return [
            [{"content": input, "role": _DEFAULT_ROLE}]
            for input in context.test_set.inputs
        ]

    raise ValueError(
        f"Invalid type for `context`: {type(context)}. Check the function's signature for allowed input types."
    )


def merge_system_messages(
    conversations: list[AnnotatedConversation],
) -> list[AnnotatedConversation]:
    """
    Merge system messages into the user/assistant messages in the conversation.

    ### Parameters
    --------------
    - `conversation`: the conversation in which to merge the system messages.

    ### Returns
    --------------
    The conversation with the system messages merged into the user messages.
    """

    merged = []
    for conversation in conversations:
        if len(conversation) < 2:
            merged.append(conversation)
            continue

        for i in range(len(conversation) - 1):
            if "role" in conversation[i] and conversation[i]["role"] == "system":
                conversation[i + 1]["content"] = (
                    conversation[i]["content"] + "\n" + conversation[i + 1]["content"]
                )

        merged.append(
            [message for message in conversation if message["role"] != "system"]
        )

    return merged
