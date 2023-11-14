import logging
import socket
from functools import cache
from typing import Any

from asyncua import Client
from asyncua import ua
from asyncua.crypto.cert_gen import setup_self_signed_certificate
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from cryptography.x509.oid import ExtendedKeyUsageOID

from .settings import get_settings
from .settings import OPCUASettings

logger = logging.getLogger('rps')


class OPCClient:
    """Wrapper for OPC UA Client.

    Args:
        settings (OPCUASettings): Settings.
    """

    def __init__(self, settings: OPCUASettings | None):
        self.settings = settings
        self._client: Client | None = None

    def __str__(self) -> str:
        return f'OPC UA {self.settings}'

    async def setup_client(self) -> Client:
        """Setups the client.

        Returns:
            Client: Returns a new client instance.
        """
        if self.settings is None:
            return None

        if self._client is not None:
            return self._client

        CERTS_DIR = get_settings().STATIC_DIR / 'certs'
        certificate = CERTS_DIR / 'cert.der'
        private_key = CERTS_DIR / 'private_key.pem'
        hostname = socket.gethostname()
        app_uri = f'urn:{hostname}:rockpaperscissors'

        await setup_self_signed_certificate(
            key_file=(CERTS_DIR / 'private_key.pem'),
            cert_file=certificate,
            app_uri=app_uri,
            host_name=hostname,
            cert_use=[ExtendedKeyUsageOID.CLIENT_AUTH],
            subject_attrs={
                'countryName': 'DE',
                'stateOrProvinceName': 'NI',
                'localityName': 'Bad Pyrmont',
                'organizationName': 'Phoenix Contact Electronics GmbH',
            },
        )

        client = Client(str(self.settings.url), timeout=2)
        client.application_uri = app_uri
        await client.set_security(
            SecurityPolicyBasic256Sha256,
            certificate=certificate,
            private_key=private_key,
        )
        self._client = client
        logger.info(self)

    async def write_value(self, v: Any, _type: ua.VariantType = ua.Int16):
        """Writes `v` to the configured node.

        Args:
            v (Any): Value to write.
            _type (ua.VariantType): Type of the node.
        """

        await self.setup_client()
        if self._client is None or self.settings is None:
            return

        try:
            async with self._client:
                node = self._client.get_node(self.settings.nodeid)
                dv = ua.DataValue(ua.Variant(v, _type))
                await node.set_value(dv)
                logger.info(f'Set "{self.settings.nodeid}" to "{v}".')
        except ua.UaError as e:
            logger.error(e)

    async def close(self):
        """Closes the client."""
        if self._client is not None:
            try:
                await self._client.close_session()
            except Exception as e:
                logger.error(e)

    async def update(self, settings: OPCUASettings | None):
        """Updates the settings. The old client will be closed.

        Args:
            settings (OPCUASettings): New settings.
        """
        if settings == self.settings:
            logger.info('OPC settings have not changed.')
            return

        logger.info('Updated OPC settings.')
        await self.close()
        self.settings = settings
        self._client = None


@cache
def get_opc_client() -> OPCClient:
    return OPCClient(get_settings().OPCUA)
