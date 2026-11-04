CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE network_nodes (
	id VARCHAR(128) NOT NULL, 
	type VARCHAR(32) NOT NULL, 
	lon FLOAT NOT NULL, 
	lat FLOAT NOT NULL, 
	geom geometry(POINT,4326) NOT NULL, 
	PRIMARY KEY (id)
);

CREATE INDEX ix_network_nodes_geom ON network_nodes USING gist (geom);

CREATE TABLE network_versions (
	id SERIAL NOT NULL, 
	hash VARCHAR(64) NOT NULL, 
	source VARCHAR(32) NOT NULL, 
	path TEXT, 
	nodes_count INTEGER NOT NULL, 
	edges_count INTEGER NOT NULL, 
	loaded_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);

CREATE INDEX ix_network_versions_hash ON network_versions (hash);

CREATE TABLE nn_models (
	id SERIAL NOT NULL, 
	name VARCHAR(128) NOT NULL, 
	version VARCHAR(64) NOT NULL, 
	architecture VARCHAR(128), 
	weights_path TEXT, 
	metrics JSONB, 
	is_stub BOOLEAN NOT NULL, 
	is_active BOOLEAN NOT NULL, 
	trained_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_nn_models_name_version UNIQUE (name, version)
);

CREATE TABLE users (
	id SERIAL NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	full_name VARCHAR(255), 
	email VARCHAR(255), 
	password_hash VARCHAR(255), 
	role VARCHAR(16) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (username), 
	UNIQUE (email)
);

CREATE TABLE road_segments (
	id VARCHAR(128) NOT NULL, 
	from_node_id VARCHAR(128) NOT NULL, 
	to_node_id VARCHAR(128) NOT NULL, 
	length FLOAT NOT NULL, 
	lanes INTEGER NOT NULL, 
	speed FLOAT NOT NULL, 
	priority INTEGER NOT NULL, 
	name VARCHAR(255), 
	road_type VARCHAR(64), 
	geom geometry(LINESTRING,4326) NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_road_segments_lanes CHECK (lanes >= 1), 
	CONSTRAINT ck_road_segments_length CHECK (length >= 0), 
	FOREIGN KEY(from_node_id) REFERENCES network_nodes (id) ON DELETE CASCADE, 
	FOREIGN KEY(to_node_id) REFERENCES network_nodes (id) ON DELETE CASCADE
);

CREATE INDEX ix_road_segments_from_node_id ON road_segments (from_node_id);

CREATE INDEX ix_road_segments_geom ON road_segments USING gist (geom);

CREATE INDEX ix_road_segments_to_node_id ON road_segments (to_node_id);

CREATE TABLE scenarios (
	id SERIAL NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	description TEXT, 
	user_id INTEGER, 
	is_base BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE SET NULL
);

CREATE TABLE agent_evaluations (
	id SERIAL NOT NULL, 
	scenario_id INTEGER, 
	config_hash VARCHAR(64) NOT NULL, 
	mode VARCHAR(16) NOT NULL, 
	evaluator VARCHAR(32) NOT NULL, 
	metrics JSONB NOT NULL, 
	duration_ms FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_agent_evaluations_mode CHECK (mode IN ('fixed', 'actuated', 'rl')), 
	FOREIGN KEY(scenario_id) REFERENCES scenarios (id) ON DELETE CASCADE
);

CREATE INDEX ix_agent_evaluations_config_hash ON agent_evaluations (config_hash);

CREATE INDEX ix_agent_evaluations_scenario_id ON agent_evaluations (scenario_id);

CREATE TABLE changes (
	id SERIAL NOT NULL, 
	scenario_id INTEGER NOT NULL, 
	seq INTEGER NOT NULL, 
	type VARCHAR(16) NOT NULL, 
	target_id VARCHAR(128) NOT NULL, 
	value JSONB, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_changes_type CHECK (type IN ('lanes', 'speed', 'close', 'open', 'traffic_light')), 
	FOREIGN KEY(scenario_id) REFERENCES scenarios (id) ON DELETE CASCADE
);

CREATE INDEX ix_changes_scenario_id ON changes (scenario_id);

CREATE TABLE forecasts (
	id SERIAL NOT NULL, 
	scenario_id INTEGER NOT NULL, 
	model_id INTEGER, 
	signal_mode VARCHAR(16) NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	changes_snapshot JSONB, 
	summary JSONB, 
	agent_metrics JSONB, 
	network_hash VARCHAR(64), 
	predictor VARCHAR(32), 
	duration_ms FLOAT, 
	error TEXT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_forecasts_signal_mode CHECK (signal_mode IN ('fixed', 'actuated', 'rl')), 
	FOREIGN KEY(scenario_id) REFERENCES scenarios (id) ON DELETE CASCADE, 
	FOREIGN KEY(model_id) REFERENCES nn_models (id) ON DELETE SET NULL
);

CREATE INDEX ix_forecasts_scenario_id ON forecasts (scenario_id);

CREATE TABLE traffic_measurements (
	id SERIAL NOT NULL, 
	edge_id VARCHAR(128) NOT NULL, 
	source VARCHAR(32) NOT NULL, 
	run_label VARCHAR(128), 
	measured_at TIMESTAMP WITH TIME ZONE, 
	interval_begin FLOAT, 
	interval_end FLOAT, 
	occupancy FLOAT, 
	mean_speed FLOAT, 
	travel_time FLOAT, 
	flow FLOAT, 
	density FLOAT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(edge_id) REFERENCES road_segments (id) ON DELETE CASCADE
);

CREATE INDEX ix_traffic_measurements_edge_id ON traffic_measurements (edge_id);

CREATE TABLE forecast_results (
	forecast_id INTEGER NOT NULL, 
	edge_id VARCHAR(128) NOT NULL, 
	occupancy FLOAT NOT NULL, 
	mean_speed FLOAT NOT NULL, 
	travel_time FLOAT, 
	is_closed BOOLEAN NOT NULL, 
	PRIMARY KEY (forecast_id, edge_id), 
	CONSTRAINT ck_forecast_results_occupancy CHECK (occupancy >= 0 AND occupancy <= 1), 
	FOREIGN KEY(forecast_id) REFERENCES forecasts (id) ON DELETE CASCADE, 
	FOREIGN KEY(edge_id) REFERENCES road_segments (id) ON DELETE CASCADE
);
