# This file contains functions for migrating data from one version of Chroma to another.
from typing import Optional, Union
from chromadb.api.client import Client
from chromadb.api.configuration import (
    Configuration,
    ConfigurationParameter,
)
from chromadb.segment.impl.vector.hnsw_params import HnswParams
from pypika import Table
from chromadb.db.base import ParameterValue, get_sql
from tqdm import tqdm


def migrate_collections(
    client: Client,
) -> None:
    """Migrates the collections in a Local Chroma instance to the latest version of ChromaDB.
    This is non-destructive and idempotent. It will only update collections that need to be updated.
    """

    # region Add CollectionConfiguration to Collections with None configuration

    # Monkeypatch the CollectionConfiguration from_json_str to handle None
    from chromadb.api.configuration import CollectionConfiguration

    original_from_json_str = CollectionConfiguration.from_json_str

    class EmptyConfiguration(Configuration):
        definitions = {}

    def patched_from_json_str(
        cls: "CollectionConfiguration", json_str: Optional[str]
    ) -> Union[EmptyConfiguration, "CollectionConfiguration"]:
        """Returns a CollectionConfiguration from the given JSON string.
        If no JSON string is provided, returns an empty configuration."""

        if json_str is None:
            return EmptyConfiguration()

        return original_from_json_str(json_str)

    # The very rare triple type hint failure
    CollectionConfiguration.from_json_str = classmethod(patched_from_json_str)  # type: ignore[method-assign, assignment, arg-type]

    # Load up the collections
    collection_models = client._server.list_collections()

    for collection_model in tqdm(collection_models, "Migrating collections"):
        # Skip collections with a configuration - this might happen if a migration was interrupted
        if not len(collection_model.configuration_json) == 0:
            continue

        # Get any existing HNSW params
        hnsw_params = HnswParams(HnswParams.extract(collection_model.metadata or {}))

        # Create ConfigurationParameters for the HNSW parameters
        space_param = ConfigurationParameter(name="space", value=hnsw_params.space)
        construction_ef_param = ConfigurationParameter(
            name="ef_construction", value=hnsw_params.construction_ef
        )
        search_ef_param = ConfigurationParameter(
            name="ef_search", value=hnsw_params.search_ef
        )
        M_param = ConfigurationParameter(name="M", value=hnsw_params.M)
        num_threads_param = ConfigurationParameter(
            name="num_threads", value=hnsw_params.num_threads
        )
        resize_factor_param = ConfigurationParameter(
            name="resize_factor", value=hnsw_params.resize_factor
        )

        # Create a new CollectionConfiguration with the HNSW parameters
        new_configuration = CollectionConfiguration(
            parameters=[
                space_param,
                construction_ef_param,
                search_ef_param,
                M_param,
                num_threads_param,
                resize_factor_param,
            ]
        )

        # Raw Dog SQL, bypassing checks since CollectionConfiguration is generally immutable
        collections_t = Table("collections")
        _sysdb = client._server._sysdb  # type: ignore[attr-defined]
        q = (
            _sysdb.querybuilder()
            .update(collections_t)
            .where(
                collections_t.id
                == ParameterValue(_sysdb.uuid_to_db(collection_model.id))
            )
        )
        q = q.set(
            collections_t.config_json_str,
            ParameterValue(new_configuration.to_json_str()),
        )
        with _sysdb.tx() as cur:
            sql, params = get_sql(q, _sysdb.parameter_format())
            cur.execute(sql, params)

    # endregion
