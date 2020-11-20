import traceback
import urllib
import urllib.parse
from pathlib import Path

from pygls.server import LanguageServer
from pygls.types import MessageType

from roosterize.interface.VSCodeInterface import VSCodeInterface
from roosterize.Utils import Utils


class RoosterizeLanguageServer(LanguageServer):
    CMD_SUGGEST_NAMING = "extension.roosterize.suggest_naming"
    CMD_DOWNLOAD_MODEL = "extension.roosterize.download_global_model"
    CMD_IMPROVE_MODEL = "extension.roosterize.improve_project_model"

    CONFIGURATION_SECTION = "RoosterizeServer"

    def __init__(self):
        super().__init__()


roosterize_server = RoosterizeLanguageServer()
ui = VSCodeInterface()


@roosterize_server.thread()
@roosterize_server.command(RoosterizeLanguageServer.CMD_SUGGEST_NAMING)
def suggest_naming(ls: LanguageServer, *args):
    ui.set_language_server(ls)
    ls.show_message("Suggesting naming...")

    paths = []
    for d in ls.workspace.documents:
        p = Path(urllib.parse.unquote_plus(urllib.parse.urlparse(d).path))
        if p.suffix != ".v":
            continue
        paths.append(p)

    if len(paths) == 0:
        ls.show_message("Please open at least one .v file!", MessageType.Error)

    for p in paths:
        try:
            ui.suggest_naming(p)
        except:
            ls.show_message_log(traceback.format_exc())
            raise


@roosterize_server.thread()
@roosterize_server.command(RoosterizeLanguageServer.CMD_DOWNLOAD_MODEL)
def download_global_model(ls: LanguageServer, *args):
    ui.set_language_server(ls)
    try:
        ui.download_global_model()
    except:
        ls.show_message_log(traceback.format_exc())
        raise


# TODO: future work
# @roosterize_server.command(RoosterizeLanguageServer.CMD_IMPROVE_MODEL)
# def improve_project_model(ls: LanguageServer, *args):
#     ls.show_message(f"From server! improve_model args: {args}")


def start_server(**options):
    tcp = Utils.get_option_as_boolean(options, "tcp", default=False)
    host = options.get("host", "127.0.0.1")
    port = options.get("port", 20145)  # Default port is for debugging

    if tcp:
        roosterize_server.start_tcp(host, port)
    else:
        roosterize_server.start_io()
