"""
Web-based Saab GlobalEye Radar Simulation using Plotly Dash.
Uses Patch() so only dynamic traces are re-sent each frame.
"""

import numpy as np
from typing import Optional
from dash import Dash, html, dcc, Input, Output, State, callback_context, Patch
import plotly.graph_objects as go

from .simulator import RadarSimulationArea
from .scenario import get_scenario, with_seed
from .replay import build_run_metadata, write_run_metadata


DARK = {'backgroundColor': '#0d0d0d', 'color': '#00ff41', 'fontFamily': 'monospace'}


def create_app(scenario_name: str = "default", seed: Optional[int] = None) -> Dash:
    scenario = with_seed(get_scenario(scenario_name), seed)
    radar_cfg = scenario.radar
    target_cfg = scenario.target
    sim_cfg = scenario.simulation
    vis_cfg = scenario.visualization

    angle_start = float(radar_cfg.coverage_angle_start_deg)
    angle_end = float(radar_cfg.coverage_angle_end_deg)
    range_km = float(radar_cfg.detection_range_km)

    n_rings = len([r for r in vis_cfg.range_rings_km if r <= range_km])
    t_illum = 1 + n_rings
    t_sweep = 2 + n_rings
    t_targets = 3 + n_rings
    t_meas = 4 + n_rings
    t_tracks = 5 + n_rings

    state = {
        "area": RadarSimulationArea(scenario=scenario),
        "frame_count": 0,
        "radar_sweep_angle": float(angle_start),
        "last_measurements": [],
    }

    def spawn_targets() -> None:
        area = state["area"]
        if len(area.targets) < target_cfg.max_targets and np.random.rand() < target_cfg.spawn_rate:
            r = np.random.uniform(50_000, range_km * 1_000)
            ang = np.radians(np.random.uniform(angle_start, angle_end))
            px, py = r * np.cos(ang), r * np.sin(ang)
            spd = np.random.uniform(target_cfg.min_velocity_ms, target_cfg.max_velocity_ms)
            va = np.radians(np.random.uniform(0, 360))
            acc = np.random.uniform(target_cfg.min_acceleration_ms2, target_cfg.max_acceleration_ms2)
            aa = np.radians(np.random.uniform(0, 360))
            area.add_target(
                px,
                py,
                spd * np.cos(va),
                spd * np.sin(va),
                acc * np.cos(aa),
                acc * np.sin(aa),
            )

    def _sweep_xy(angle_start_deg: float, width_deg: float, n: int = 20):
        t = np.linspace(np.radians(angle_start_deg), np.radians(angle_start_deg + width_deg), n)
        return (
            np.concatenate([[0], range_km * np.cos(t), [0]]).tolist(),
            np.concatenate([[0], range_km * np.sin(t), [0]]).tolist(),
        )

    def build_initial_figure():
        fig = go.Figure()

        theta = np.linspace(np.radians(angle_start), np.radians(angle_end), 120)
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], range_km * np.cos(theta), [0]]).tolist(),
            y=np.concatenate([[0], range_km * np.sin(theta), [0]]).tolist(),
            fill='toself', fillcolor='rgba(0,40,0,0.35)',
            line=dict(color='rgba(0,180,0,0.5)', width=1.5, dash='dot'),
            name='Coverage', hoverinfo='skip'
        ))

        for r_km in [r for r in vis_cfg.range_rings_km if r <= range_km]:
            t = np.linspace(np.radians(angle_start), np.radians(angle_end), 120)
            fig.add_trace(go.Scatter(
                x=(r_km * np.cos(t)).tolist(), y=(r_km * np.sin(t)).tolist(),
                mode='lines',
                line=dict(color='rgba(0,150,0,0.25)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False
            ))

        ix, iy = _sweep_xy(state["radar_sweep_angle"], vis_cfg.radar_sweep_width_deg * 2.5, 30)
        fig.add_trace(go.Scatter(
            x=ix, y=iy, fill='toself',
            fillcolor=f'rgba(0,255,80,{vis_cfg.illumination_alpha + 0.15})',
            line=dict(color='rgba(100,255,100,0.6)', width=1),
            name='Active Beam', hoverinfo='skip'
        ))

        sx, sy = _sweep_xy(state["radar_sweep_angle"], vis_cfg.radar_sweep_width_deg, 20)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, fill='toself',
            fillcolor=f'rgba(0,200,0,{vis_cfg.radar_sweep_alpha})',
            line=dict(color='rgba(0,0,0,0)', width=0),
            hoverinfo='skip', showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=10, color='deepskyblue', symbol='circle', line=dict(width=1, color='white')),
            name='True Targets', hoverinfo='text'
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=8, color='yellow', symbol='x', line=dict(width=2, color='yellow')),
            name='Detections', hoverinfo='name'
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=12, color='red', symbol='square', line=dict(width=2, color='white')),
            name='Tracked', hoverinfo='text'
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=14, color='lime', symbol='star', line=dict(width=2, color='white')),
            name='GlobalEye', hoverinfo='name'
        ))

        mid_ang = np.radians((angle_start + angle_end) / 2)
        ring_annotations = [
            dict(
                x=r_km * np.cos(mid_ang) * 0.98,
                y=r_km * np.sin(mid_ang) * 0.98,
                text=f'{r_km} km',
                showarrow=False,
                font=dict(color='rgba(0,200,0,0.6)', size=10),
                xref='x', yref='y'
            )
            for r_km in [r for r in vis_cfg.range_rings_km if r <= range_km]
        ]

        fig.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1a0d',
            font=dict(color='#00ff41', family='monospace'),
            title=dict(text='GlobalEye Radar', font=dict(color='#00ff41', size=14)),
            xaxis=dict(
                title='X (km)', range=list(vis_cfg.plot_xlim_km(range_km)), color='#00aa33',
                showgrid=True, gridcolor='rgba(0,100,0,0.2)', zeroline=False,
                tickfont=dict(color='#00aa33'), scaleanchor='y', scaleratio=1
            ),
            yaxis=dict(
                title='Y (km)', range=list(vis_cfg.plot_ylim_km_for_range(range_km)), color='#00aa33',
                showgrid=True, gridcolor='rgba(0,100,0,0.2)', zeroline=False,
                tickfont=dict(color='#00aa33')
            ),
            legend=dict(bgcolor='rgba(0,20,0,0.7)', bordercolor='#00aa33', borderwidth=1, font=dict(color='#00ff41')),
            hovermode='closest',
            margin=dict(l=60, r=60, t=50, b=50),
            uirevision='constant',
            annotations=ring_annotations,
        )
        return fig

    def build_patch():
        p = Patch()
        area = state["area"]
        confirmed = area.tracker.get_confirmed_tracks()

        ix, iy = _sweep_xy(state["radar_sweep_angle"], vis_cfg.radar_sweep_width_deg * 2.5, 30)
        p['data'][t_illum]['x'] = ix
        p['data'][t_illum]['y'] = iy

        sx, sy = _sweep_xy(state["radar_sweep_angle"], vis_cfg.radar_sweep_width_deg, 20)
        p['data'][t_sweep]['x'] = sx
        p['data'][t_sweep]['y'] = sy

        if area.targets:
            tp = np.array([t.position / 1000.0 for t in area.targets])
            tv = [np.linalg.norm(t.velocity) * 3.6 for t in area.targets]
            p['data'][t_targets]['x'] = tp[:, 0].tolist()
            p['data'][t_targets]['y'] = tp[:, 1].tolist()
            p['data'][t_targets]['text'] = [
                f'Target {i}<br>{tp[i,0]:.1f}, {tp[i,1]:.1f} km<br>{tv[i]:.0f} km/h'
                for i in range(len(area.targets))
            ]
        else:
            p['data'][t_targets]['x'] = []
            p['data'][t_targets]['y'] = []

        if state["last_measurements"]:
            mp = np.array([m[1] / 1000.0 for m in state["last_measurements"]])
            p['data'][t_meas]['x'] = mp[:, 0].tolist()
            p['data'][t_meas]['y'] = mp[:, 1].tolist()
        else:
            p['data'][t_meas]['x'] = []
            p['data'][t_meas]['y'] = []

        if confirmed:
            tp2 = np.array([t.get_estimated_position() / 1000.0 for t in confirmed])
            p['data'][t_tracks]['x'] = tp2[:, 0].tolist()
            p['data'][t_tracks]['y'] = tp2[:, 1].tolist()
            p['data'][t_tracks]['text'] = [
                f'Track {t.id}<br>{tp2[i,0]:.1f}, {tp2[i,1]:.1f} km<br>'
                f'{np.linalg.norm(t.get_estimated_velocity()) * 3.6:.0f} km/h<br>'
                f'Age: {t.time_since_update:.1f}s'
                for i, t in enumerate(confirmed)
            ]
        else:
            p['data'][t_tracks]['x'] = []
            p['data'][t_tracks]['y'] = []

        p['layout']['title']['text'] = (
            f'GlobalEye Radar  ·  Frame {state["frame_count"]}  ·  '
            f'Targets: {len(area.targets)}  ·  Tracks: {len(confirmed)}'
        )

        mid_ang = np.radians((angle_start + angle_end) / 2)
        ring_anns = [
            dict(
                x=r_km * np.cos(mid_ang) * 0.98,
                y=r_km * np.sin(mid_ang) * 0.98,
                text=f'{r_km} km', showarrow=False,
                font=dict(color='rgba(0,200,0,0.6)', size=10), xref='x', yref='y'
            )
            for r_km in [r for r in vis_cfg.range_rings_km if r <= range_km]
        ]

        vel_anns = []
        for track in confirmed:
            pos = track.get_estimated_position() / 1000.0
            vel = track.get_estimated_velocity()
            end = pos + vel * vis_cfg.velocity_vector_time_s / 1000.0
            vel_anns.append(dict(
                x=end[0], y=end[1], ax=pos[0], ay=pos[1],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3, arrowsize=1,
                arrowwidth=2, arrowcolor='rgba(255,80,80,0.8)', opacity=0.8
            ))

        p['layout']['annotations'] = ring_anns + vel_anns
        return p

    def make_stats():
        area = state["area"]
        confirmed = area.tracker.get_confirmed_tracks()
        items = [
            ('Frame', str(state["frame_count"])),
            ('Sim Time', f'{area.time:.0f}s'),
            ('Targets', str(len(area.targets))),
            ('Tracks', str(len(confirmed))),
            ('Range', f'{range_km:.0f} km'),
            ('Coverage', f'{radar_cfg.coverage_angle_total_deg:.0f}°'),
            ('Pd', f'{radar_cfg.detection_probability * 100:.0f}%'),
            ('Noise', f'{radar_cfg.measurement_noise_m:.0f} m'),
            ('Sweep', f'{state["radar_sweep_angle"]:.1f}°'),
        ]
        return [html.Span([html.Strong(k + ': '), v], style={'marginRight': '8px'}) for k, v in items]

    app = Dash(__name__)
    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True

    app.layout = html.Div(style={**DARK, 'padding': '16px'}, children=[
        html.H2('Saab GlobalEye  ·  Kalman Filter Tracking',
                style={'textAlign': 'center', 'color': '#00ff41', 'letterSpacing': '2px', 'marginBottom': '12px'}),
        html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'marginBottom': '12px', 'flexWrap': 'wrap'}, children=[
            html.Button('⏸  Pause', id='btn-pause', n_clicks=0,
                        style={'padding': '8px 20px', 'backgroundColor': '#003300', 'color': '#00ff41', 'border': '1px solid #00ff41', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
            html.Button('🔄  Reset', id='btn-reset', n_clicks=0,
                        style={'padding': '8px 20px', 'backgroundColor': '#330000', 'color': '#ff4444', 'border': '1px solid #ff4444', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
            html.Span('Interval:', style={'marginLeft': '16px'}),
            html.Div(dcc.Slider(id='speed-slider', min=100, max=2000, step=100, value=sim_cfg.frame_interval_ms,
                                marks={100: '100ms', 500: '500ms', 1000: '1s', 2000: '2s'},
                                tooltip={'placement': 'bottom', 'always_visible': True}),
                     style={'width': '300px'}),
        ]),
        dcc.Graph(id='radar-graph', figure=build_initial_figure(), style={'height': '75vh'}, config={'displayModeBar': True}),
        html.Div(id='stats-bar',
                 style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #00aa33',
                        'borderRadius': '4px', 'backgroundColor': '#050f05', 'fontSize': '13px',
                        'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap'}),
        dcc.Interval(id='interval', interval=sim_cfg.frame_interval_ms, n_intervals=0),
        dcc.Store(id='store', data={'running': True}),
    ])

    @app.callback(
        Output('store', 'data'),
        Output('btn-pause', 'children'),
        Input('btn-pause', 'n_clicks'),
        Input('btn-reset', 'n_clicks'),
        State('store', 'data'),
        prevent_initial_call=True,
    )
    def handle_buttons(_p, _r, store):
        tid = callback_context.triggered_id
        if tid == 'btn-reset':
            state["area"] = RadarSimulationArea(scenario=scenario)
            state["frame_count"] = 0
            state["radar_sweep_angle"] = float(angle_start)
            state["last_measurements"] = []
            store['running'] = True
            return store, '⏸  Pause'
        if tid == 'btn-pause':
            store['running'] = not store['running']
        label = '▶  Resume' if not store['running'] else '⏸  Pause'
        return store, label

    @app.callback(Output('interval', 'interval'), Input('speed-slider', 'value'))
    def set_speed(val):
        return val

    @app.callback(
        Output('radar-graph', 'figure'),
        Output('stats-bar', 'children'),
        Input('interval', 'n_intervals'),
        State('store', 'data'),
    )
    def tick(_, store):
        if store and not store.get('running', True):
            return build_patch(), make_stats()

        sweep_range = angle_end - angle_start
        state["radar_sweep_angle"] = angle_start + (
            (state["radar_sweep_angle"] - angle_start + vis_cfg.radar_sweep_speed_deg_per_frame)
            % sweep_range
        )

        spawn_targets()
        state["area"].step()
        state["last_measurements"] = state["area"].last_measurements
        state["frame_count"] += 1

        return build_patch(), make_stats()

    return app


def main(scenario_name: str = "default", seed: Optional[int] = None, metadata_out: Optional[str] = None) -> None:
    scenario = with_seed(get_scenario(scenario_name), seed)
    app = create_app(scenario_name=scenario_name, seed=seed)
    print('=' * 60)
    print('Saab GlobalEye Web Radar  ->  http://localhost:8050')
    print('=' * 60)
    metadata = build_run_metadata(mode="web", scenario=scenario, seed_override=seed)
    if metadata_out:
        write_run_metadata(metadata_out, metadata)
        print(f"Run metadata written to: {metadata_out}")
    app.run(debug=True, port=8050, host='0.0.0.0')


if __name__ == '__main__':
    main()
