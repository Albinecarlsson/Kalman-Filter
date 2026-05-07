"""
Web-based Saab GlobalEye Radar Simulation using Plotly Dash.
Uses Patch() so only dynamic traces are re-sent each frame.
"""

import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context, Patch
import plotly.graph_objects as go

from simulator import RadarSimulationArea
from config import RadarConfig, TargetConfig, SimulationConfig, VisualizationConfig


# ── Constants ─────────────────────────────────────────────────────────────────
ANGLE_START = float(RadarConfig.COVERAGE_ANGLE_START_DEG)
ANGLE_END   = float(RadarConfig.COVERAGE_ANGLE_START_DEG + RadarConfig.COVERAGE_ANGLE_TOTAL_DEG)
RANGE_KM    = float(RadarConfig.DETECTION_RANGE_KM)

# Fixed trace indices in the figure (static ones are never re-sent)
N_RINGS     = len([r for r in VisualizationConfig.RANGE_RINGS_KM if r <= RANGE_KM])
T_COVERAGE  = 0
T_RINGS     = list(range(1, 1 + N_RINGS))
T_ILLUM     = 1 + N_RINGS
T_SWEEP     = 2 + N_RINGS
T_TARGETS   = 3 + N_RINGS
T_MEAS      = 4 + N_RINGS
T_TRACKS    = 5 + N_RINGS
T_PLATFORM  = 6 + N_RINGS

# ── Globals (server-side state) ──────────────────────────────────────────────
area = RadarSimulationArea()
frame_count = 0
radar_sweep_angle = float(ANGLE_START)
last_measurements = []


def spawn_targets():
    if len(area.targets) < TargetConfig.MAX_TARGETS:
        if np.random.rand() < TargetConfig.SPAWN_RATE:
            r   = np.random.uniform(50_000, RANGE_KM * 1_000)
            ang = np.radians(np.random.uniform(ANGLE_START, ANGLE_END))
            px, py = r * np.cos(ang), r * np.sin(ang)
            spd = np.random.uniform(TargetConfig.MIN_VELOCITY_MS, TargetConfig.MAX_VELOCITY_MS)
            va  = np.radians(np.random.uniform(0, 360))
            acc = np.random.uniform(TargetConfig.MIN_ACCELERATION_MS2, TargetConfig.MAX_ACCELERATION_MS2)
            aa  = np.radians(np.random.uniform(0, 360))
            area.add_target(px, py,
                            spd * np.cos(va), spd * np.sin(va),
                            acc * np.cos(aa), acc * np.sin(aa))


def _sweep_xy(angle_start_deg, width_deg, n=20):
    """Return (x, y) arrays for a wedge starting at angle_start_deg."""
    t = np.linspace(np.radians(angle_start_deg),
                    np.radians(angle_start_deg + width_deg), n)
    return (np.concatenate([[0], RANGE_KM * np.cos(t), [0]]).tolist(),
            np.concatenate([[0], RANGE_KM * np.sin(t), [0]]).tolist())


def build_initial_figure():
    """Build the full figure once — static traces + empty dynamic placeholders."""
    fig = go.Figure()

    # T_COVERAGE — coverage sector
    theta = np.linspace(np.radians(ANGLE_START), np.radians(ANGLE_END), 120)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0], RANGE_KM * np.cos(theta), [0]]).tolist(),
        y=np.concatenate([[0], RANGE_KM * np.sin(theta), [0]]).tolist(),
        fill='toself', fillcolor='rgba(0,40,0,0.35)',
        line=dict(color='rgba(0,180,0,0.5)', width=1.5, dash='dot'),
        name='Coverage', hoverinfo='skip'
    ))

    # T_RINGS — range rings (static)
    for r_km in [r for r in VisualizationConfig.RANGE_RINGS_KM if r <= RANGE_KM]:
        t = np.linspace(np.radians(ANGLE_START), np.radians(ANGLE_END), 120)
        fig.add_trace(go.Scatter(
            x=(r_km * np.cos(t)).tolist(), y=(r_km * np.sin(t)).tolist(),
            mode='lines',
            line=dict(color='rgba(0,150,0,0.25)', width=1, dash='dot'),
            hoverinfo='skip', showlegend=False
        ))

    # T_ILLUM — illumination beam (dynamic placeholder)
    ix, iy = _sweep_xy(radar_sweep_angle, VisualizationConfig.RADAR_SWEEP_WIDTH_DEG * 2.5, 30)
    fig.add_trace(go.Scatter(
        x=ix, y=iy, fill='toself',
        fillcolor=f'rgba(0,255,80,{VisualizationConfig.ILLUMINATION_ALPHA + 0.15})',
        line=dict(color='rgba(100,255,100,0.6)', width=1),
        name='Active Beam', hoverinfo='skip'
    ))

    # T_SWEEP — sweep trail (dynamic placeholder)
    sx, sy = _sweep_xy(radar_sweep_angle, VisualizationConfig.RADAR_SWEEP_WIDTH_DEG, 20)
    fig.add_trace(go.Scatter(
        x=sx, y=sy, fill='toself',
        fillcolor=f'rgba(0,200,0,{VisualizationConfig.RADAR_SWEEP_ALPHA})',
        line=dict(color='rgba(0,0,0,0)', width=0),
        hoverinfo='skip', showlegend=False
    ))

    # T_TARGETS — true targets (dynamic placeholder)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers',
        marker=dict(size=10, color='deepskyblue', symbol='circle',
                    line=dict(width=1, color='white')),
        name='True Targets', hoverinfo='text'
    ))

    # T_MEAS — detections (dynamic placeholder)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers',
        marker=dict(size=8, color='yellow', symbol='x',
                    line=dict(width=2, color='yellow')),
        name='Detections', hoverinfo='name'
    ))

    # T_TRACKS — confirmed tracks (dynamic placeholder)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers',
        marker=dict(size=12, color='red', symbol='square',
                    line=dict(width=2, color='white')),
        name='Tracked', hoverinfo='text'
    ))

    # T_PLATFORM — radar platform (static)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=14, color='lime', symbol='star',
                    line=dict(width=2, color='white')),
        name='GlobalEye', hoverinfo='name'
    ))

    # Range-ring labels (static annotations)
    ring_annotations = []
    mid_ang = np.radians((ANGLE_START + ANGLE_END) / 2)
    for r_km in [r for r in VisualizationConfig.RANGE_RINGS_KM if r <= RANGE_KM]:
        ring_annotations.append(dict(
            x=r_km * np.cos(mid_ang) * 0.98,
            y=r_km * np.sin(mid_ang) * 0.98,
            text=f'{r_km} km', showarrow=False,
            font=dict(color='rgba(0,200,0,0.6)', size=10),
            xref='x', yref='y'
        ))

    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1a0d',
        font=dict(color='#00ff41', family='monospace'),
        title=dict(text='GlobalEye Radar', font=dict(color='#00ff41', size=14)),
        xaxis=dict(title='X (km)', range=list(VisualizationConfig.PLOT_XLIM_KM),
                   color='#00aa33', showgrid=True, gridcolor='rgba(0,100,0,0.2)',
                   zeroline=False, tickfont=dict(color='#00aa33'),
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(title='Y (km)', range=list(VisualizationConfig.PLOT_YLIM_KM),
                   color='#00aa33', showgrid=True, gridcolor='rgba(0,100,0,0.2)',
                   zeroline=False, tickfont=dict(color='#00aa33')),
        legend=dict(bgcolor='rgba(0,20,0,0.7)', bordercolor='#00aa33',
                    borderwidth=1, font=dict(color='#00ff41')),
        hovermode='closest',
        margin=dict(l=60, r=60, t=50, b=50),
        uirevision='constant',
        annotations=ring_annotations
    )
    return fig


def build_patch():
    """Return a Patch that updates only the dynamic traces and annotations."""
    p = Patch()
    confirmed = area.tracker.get_confirmed_tracks()

    # Sweep beam
    ix, iy = _sweep_xy(radar_sweep_angle, VisualizationConfig.RADAR_SWEEP_WIDTH_DEG * 2.5, 30)
    p['data'][T_ILLUM]['x'] = ix
    p['data'][T_ILLUM]['y'] = iy

    sx, sy = _sweep_xy(radar_sweep_angle, VisualizationConfig.RADAR_SWEEP_WIDTH_DEG, 20)
    p['data'][T_SWEEP]['x'] = sx
    p['data'][T_SWEEP]['y'] = sy

    # True targets
    if area.targets:
        tp = np.array([t.position / 1000.0 for t in area.targets])
        tv = [np.linalg.norm(t.velocity) * 3.6 for t in area.targets]
        p['data'][T_TARGETS]['x'] = tp[:, 0].tolist()
        p['data'][T_TARGETS]['y'] = tp[:, 1].tolist()
        p['data'][T_TARGETS]['text'] = [
            f'Target {i}<br>{tp[i,0]:.1f}, {tp[i,1]:.1f} km<br>{tv[i]:.0f} km/h'
            for i in range(len(area.targets))
        ]
    else:
        p['data'][T_TARGETS]['x'] = []
        p['data'][T_TARGETS]['y'] = []

    # Detections
    if last_measurements:
        mp = np.array([m[1] / 1000.0 for m in last_measurements])
        p['data'][T_MEAS]['x'] = mp[:, 0].tolist()
        p['data'][T_MEAS]['y'] = mp[:, 1].tolist()
    else:
        p['data'][T_MEAS]['x'] = []
        p['data'][T_MEAS]['y'] = []

    # Tracks
    if confirmed:
        tp2 = np.array([t.get_estimated_position() / 1000.0 for t in confirmed])
        p['data'][T_TRACKS]['x'] = tp2[:, 0].tolist()
        p['data'][T_TRACKS]['y'] = tp2[:, 1].tolist()
        p['data'][T_TRACKS]['text'] = [
            f'Track {t.id}<br>{tp2[i,0]:.1f}, {tp2[i,1]:.1f} km<br>'
            f'{np.linalg.norm(t.get_estimated_velocity()) * 3.6:.0f} km/h<br>'
            f'Age: {t.time_since_update:.1f}s'
            for i, t in enumerate(confirmed)
        ]
    else:
        p['data'][T_TRACKS]['x'] = []
        p['data'][T_TRACKS]['y'] = []

    # Title
    p['layout']['title']['text'] = (
        f'GlobalEye Radar  ·  Frame {frame_count}  ·  '
        f'Targets: {len(area.targets)}  ·  Tracks: {len(confirmed)}'
    )

    # Velocity vector annotations (ring labels are index 0..N_RINGS-1, keep them)
    n_ring_labels = N_RINGS
    mid_ang = np.radians((ANGLE_START + ANGLE_END) / 2)
    ring_anns = [dict(
        x=r_km * np.cos(mid_ang) * 0.98, y=r_km * np.sin(mid_ang) * 0.98,
        text=f'{r_km} km', showarrow=False,
        font=dict(color='rgba(0,200,0,0.6)', size=10), xref='x', yref='y'
    ) for r_km in [r for r in VisualizationConfig.RANGE_RINGS_KM if r <= RANGE_KM]]

    vel_anns = []
    for track in confirmed:
        pos = track.get_estimated_position() / 1000.0
        vel = track.get_estimated_velocity()
        end = pos + vel * VisualizationConfig.VELOCITY_VECTOR_TIME_S / 1000.0
        vel_anns.append(dict(
            x=end[0], y=end[1], ax=pos[0], ay=pos[1],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=2, arrowcolor='rgba(255,80,80,0.8)', opacity=0.8
        ))

    p['layout']['annotations'] = ring_anns + vel_anns
    return p


# ── Dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

DARK = {'backgroundColor': '#0d0d0d', 'color': '#00ff41', 'fontFamily': 'monospace'}

app.layout = html.Div(style={**DARK, 'padding': '16px'}, children=[
    html.H2('Saab GlobalEye  ·  Kalman Filter Tracking',
            style={'textAlign': 'center', 'color': '#00ff41', 'letterSpacing': '2px',
                   'marginBottom': '12px'}),

    html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center',
                    'marginBottom': '12px', 'flexWrap': 'wrap'}, children=[
        html.Button('⏸  Pause', id='btn-pause', n_clicks=0,
                    style={'padding': '8px 20px', 'backgroundColor': '#003300',
                           'color': '#00ff41', 'border': '1px solid #00ff41',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
        html.Button('🔄  Reset', id='btn-reset', n_clicks=0,
                    style={'padding': '8px 20px', 'backgroundColor': '#330000',
                           'color': '#ff4444', 'border': '1px solid #ff4444',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
        html.Span('Interval:', style={'marginLeft': '16px'}),
        html.Div(dcc.Slider(id='speed-slider', min=100, max=2000, step=100,
                            value=SimulationConfig.FRAME_INTERVAL_MS,
                            marks={100: '100ms', 500: '500ms', 1000: '1s', 2000: '2s'},
                            tooltip={'placement': 'bottom', 'always_visible': True}),
                 style={'width': '300px'}),
    ]),

    dcc.Graph(id='radar-graph',
              figure=build_initial_figure(),
              style={'height': '75vh'},
              config={'displayModeBar': True}),

    html.Div(id='stats-bar',
             style={'marginTop': '10px', 'padding': '10px',
                    'border': '1px solid #00aa33', 'borderRadius': '4px',
                    'backgroundColor': '#050f05', 'fontSize': '13px',
                    'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap'}),

    dcc.Interval(id='interval', interval=SimulationConfig.FRAME_INTERVAL_MS, n_intervals=0),
    dcc.Store(id='store', data={'running': True}),
])


@app.callback(
    Output('store', 'data'),
    Output('btn-pause', 'children'),
    Input('btn-pause', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    State('store', 'data'),
    prevent_initial_call=True
)
def handle_buttons(_p, _r, store):
    global area, frame_count, radar_sweep_angle, last_measurements
    tid = callback_context.triggered_id
    if tid == 'btn-reset':
        area = RadarSimulationArea()
        frame_count = 0
        radar_sweep_angle = float(ANGLE_START)
        last_measurements = []
        store['running'] = True
        return store, '⏸  Pause'
    if tid == 'btn-pause':
        store['running'] = not store['running']
    label = '▶  Resume' if not store['running'] else '⏸  Pause'
    return store, label


@app.callback(
    Output('interval', 'interval'),
    Input('speed-slider', 'value')
)
def set_speed(val):
    return val


@app.callback(
    Output('radar-graph', 'figure'),
    Output('stats-bar', 'children'),
    Input('interval', 'n_intervals'),
    State('store', 'data')
)
def tick(_, store):
    global frame_count, radar_sweep_angle, last_measurements

    if store and not store.get('running', True):
        return build_patch(), make_stats()

    # Advance sweep
    sweep_range = ANGLE_END - ANGLE_START
    radar_sweep_angle = ANGLE_START + (
        (radar_sweep_angle - ANGLE_START + VisualizationConfig.RADAR_SWEEP_SPEED_DEG_PER_FRAME)
        % sweep_range
    )

    spawn_targets()
    area.step()
    last_measurements = area.radar.detect(area.targets)
    frame_count += 1

    return build_patch(), make_stats()


def make_stats():
    confirmed = area.tracker.get_confirmed_tracks()
    items = [
        ('Frame', str(frame_count)),
        ('Sim Time', f'{area.time:.0f}s'),
        ('Targets', str(len(area.targets))),
        ('Tracks', str(len(confirmed))),
        ('Range', f'{RANGE_KM:.0f} km'),
        ('Coverage', f'{RadarConfig.COVERAGE_ANGLE_TOTAL_DEG:.0f}°'),
        ('Pd', f'{RadarConfig.DETECTION_PROBABILITY * 100:.0f}%'),
        ('Noise', f'{RadarConfig.MEASUREMENT_NOISE_M:.0f} m'),
        ('Sweep', f'{radar_sweep_angle:.1f}°'),
    ]
    return [
        html.Span([html.Strong(k + ': '), v], style={'marginRight': '8px'})
        for k, v in items
    ]


if __name__ == '__main__':
    print('=' * 60)
    print('Saab GlobalEye Web Radar  ->  http://localhost:8050')
    print('=' * 60)
    app.run(debug=True, port=8050, host='0.0.0.0')



def spawn_targets():
    """Spawn random targets into the simulation area."""
    if len(area.targets) < TargetConfig.MAX_TARGETS:
        if np.random.rand() < TargetConfig.SPAWN_RATE:
            r   = np.random.uniform(50_000, RANGE_KM * 1_000)
            ang = np.radians(np.random.uniform(ANGLE_START, ANGLE_END))
            px, py = r * np.cos(ang), r * np.sin(ang)

            spd = np.random.uniform(TargetConfig.MIN_VELOCITY_MS, TargetConfig.MAX_VELOCITY_MS)
            va  = np.radians(np.random.uniform(0, 360))
            vx, vy = spd * np.cos(va), spd * np.sin(va)

            acc = np.random.uniform(TargetConfig.MIN_ACCELERATION_MS2, TargetConfig.MAX_ACCELERATION_MS2)
            aa  = np.radians(np.random.uniform(0, 360))
            ax, ay = acc * np.cos(aa), acc * np.sin(aa)

            area.add_target(px, py, vx, vy, ax, ay)


def build_figure():
    """Build and return the full Plotly radar figure."""
    fig = go.Figure()

    # ── Coverage sector ───────────────────────────────────────────────────────
    theta = np.linspace(np.radians(ANGLE_START), np.radians(ANGLE_END), 120)
    xs = np.concatenate([[0], RANGE_KM * np.cos(theta), [0]])
    ys = np.concatenate([[0], RANGE_KM * np.sin(theta), [0]])
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill='toself', fillcolor='rgba(0,40,0,0.35)',
        line=dict(color='rgba(0,180,0,0.5)', width=1.5, dash='dot'),
        name='Coverage', hoverinfo='skip'
    ))

    # ── Range rings ───────────────────────────────────────────────────────────
    for r_km in VisualizationConfig.RANGE_RINGS_KM:
        if r_km <= RANGE_KM:
            t = np.linspace(np.radians(ANGLE_START), np.radians(ANGLE_END), 120)
            fig.add_trace(go.Scatter(
                x=r_km * np.cos(t), y=r_km * np.sin(t),
                mode='lines',
                line=dict(color='rgba(0,150,0,0.25)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False
            ))
            mid_ang = np.radians((ANGLE_START + ANGLE_END) / 2)
            fig.add_annotation(
                x=r_km * np.cos(mid_ang) * 0.98,
                y=r_km * np.sin(mid_ang) * 0.98,
                text=f'{r_km} km',
                showarrow=False,
                font=dict(color='rgba(0,200,0,0.6)', size=10),
                xref='x', yref='y'
            )

    # ── Illumination beam (bright primary) ────────────────────────────────────
    beam_end = radar_sweep_angle + VisualizationConfig.RADAR_SWEEP_WIDTH_DEG * 2.5
    t_ill = np.linspace(np.radians(radar_sweep_angle), np.radians(beam_end), 30)
    xi = np.concatenate([[0], RANGE_KM * np.cos(t_ill), [0]])
    yi = np.concatenate([[0], RANGE_KM * np.sin(t_ill), [0]])
    fig.add_trace(go.Scatter(
        x=xi, y=yi,
        fill='toself',
        fillcolor=f'rgba(0,255,80,{VisualizationConfig.ILLUMINATION_ALPHA + 0.15})',
        line=dict(color='rgba(100,255,100,0.6)', width=1),
        name='Active Beam', hoverinfo='skip'
    ))

    # ── Sweep trail ───────────────────────────────────────────────────────────
    trail_end = radar_sweep_angle + VisualizationConfig.RADAR_SWEEP_WIDTH_DEG
    t_sw = np.linspace(np.radians(radar_sweep_angle), np.radians(trail_end), 20)
    xs2 = np.concatenate([[0], RANGE_KM * np.cos(t_sw), [0]])
    ys2 = np.concatenate([[0], RANGE_KM * np.sin(t_sw), [0]])
    fig.add_trace(go.Scatter(
        x=xs2, y=ys2,
        fill='toself',
        fillcolor=f'rgba(0,200,0,{VisualizationConfig.RADAR_SWEEP_ALPHA})',
        line=dict(color='rgba(0,0,0,0)', width=0),
        hoverinfo='skip', showlegend=False
    ))

    # ── True targets ──────────────────────────────────────────────────────────
    if area.targets:
        tp = np.array([t.position / 1000.0 for t in area.targets])
        tv = np.array([np.linalg.norm(t.velocity) * 3.6 for t in area.targets])
        hover = [f'Target {i}<br>{tp[i,0]:.1f}, {tp[i,1]:.1f} km<br>{tv[i]:.0f} km/h'
                 for i in range(len(area.targets))]
        fig.add_trace(go.Scatter(
            x=tp[:, 0], y=tp[:, 1],
            mode='markers',
            marker=dict(size=10, color='deepskyblue', symbol='circle',
                        line=dict(width=1, color='white')),
            name='True Targets', text=hover, hoverinfo='text'
        ))

    # ── Radar detections ──────────────────────────────────────────────────────
    measurements = area.radar.detect(area.targets)
    if measurements:
        mp = np.array([m[1] / 1000.0 for m in measurements])
        fig.add_trace(go.Scatter(
            x=mp[:, 0], y=mp[:, 1],
            mode='markers',
            marker=dict(size=8, color='yellow', symbol='x',
                        line=dict(width=2, color='yellow')),
            name='Detections', hoverinfo='name'
        ))

    # ── Confirmed tracks + velocity vectors ───────────────────────────────────
    confirmed = area.tracker.get_confirmed_tracks()
    if confirmed:
        tp2 = np.array([t.get_estimated_position() / 1000.0 for t in confirmed])
        hover2 = [
            f'Track {t.id}<br>'
            f'{tp2[i,0]:.1f}, {tp2[i,1]:.1f} km<br>'
            f'{np.linalg.norm(t.get_estimated_velocity()) * 3.6:.0f} km/h<br>'
            f'Age: {t.time_since_update:.1f}s'
            for i, t in enumerate(confirmed)
        ]
        fig.add_trace(go.Scatter(
            x=tp2[:, 0], y=tp2[:, 1],
            mode='markers',
            marker=dict(size=12, color='red', symbol='square',
                        line=dict(width=2, color='white')),
            name='Tracked', text=hover2, hoverinfo='text'
        ))

        for track in confirmed:
            pos = track.get_estimated_position() / 1000.0
            vel = track.get_estimated_velocity()
            end = pos + vel * VisualizationConfig.VELOCITY_VECTOR_TIME_S / 1000.0
            fig.add_annotation(
                x=end[0], y=end[1],
                ax=pos[0], ay=pos[1],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3, arrowsize=1,
                arrowwidth=2, arrowcolor='rgba(255,80,80,0.8)',
                opacity=0.8
            )

    # ── Radar platform ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=14, color='lime', symbol='star',
                    line=dict(width=2, color='white')),
        name='GlobalEye', hoverinfo='name'
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0d1a0d',
        font=dict(color='#00ff41', family='monospace'),
        title=dict(
            text=(f'GlobalEye Radar  ·  Frame {frame_count}  ·  '
                  f'Targets: {len(area.targets)}  ·  '
                  f'Tracks: {len(confirmed)}'),
            font=dict(color='#00ff41', size=14)
        ),
        xaxis=dict(
            title='X (km)',
            range=list(VisualizationConfig.PLOT_XLIM_KM),
            color='#00aa33',
            showgrid=True, gridcolor='rgba(0,100,0,0.2)',
            zeroline=False, tickfont=dict(color='#00aa33'),
            scaleanchor='y', scaleratio=1
        ),
        yaxis=dict(
            title='Y (km)',
            range=list(VisualizationConfig.PLOT_YLIM_KM),
            color='#00aa33',
            showgrid=True, gridcolor='rgba(0,100,0,0.2)',
            zeroline=False, tickfont=dict(color='#00aa33')
        ),
        legend=dict(
            bgcolor='rgba(0,20,0,0.7)', bordercolor='#00aa33',
            borderwidth=1, font=dict(color='#00ff41')
        ),
        hovermode='closest',
        margin=dict(l=60, r=60, t=50, b=50),
        uirevision='constant'
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

DARK = {'backgroundColor': '#0d0d0d', 'color': '#00ff41', 'fontFamily': 'monospace'}

app.layout = html.Div(style={**DARK, 'padding': '16px'}, children=[
    html.H2('Saab GlobalEye  ·  Kalman Filter Tracking',
            style={'textAlign': 'center', 'color': '#00ff41', 'letterSpacing': '2px',
                   'marginBottom': '12px'}),

    html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center',
                    'marginBottom': '12px', 'flexWrap': 'wrap'}, children=[
        html.Button('⏸  Pause', id='btn-pause', n_clicks=0,
                    style={'padding': '8px 20px', 'backgroundColor': '#003300',
                           'color': '#00ff41', 'border': '1px solid #00ff41',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
        html.Button('🔄  Reset', id='btn-reset', n_clicks=0,
                    style={'padding': '8px 20px', 'backgroundColor': '#330000',
                           'color': '#ff4444', 'border': '1px solid #ff4444',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '14px'}),
        html.Span('Interval:', style={'marginLeft': '16px'}),
        html.Div(dcc.Slider(id='speed-slider', min=100, max=2000, step=100,
                   value=SimulationConfig.FRAME_INTERVAL_MS,
                   marks={100: '100ms', 500: '500ms', 1000: '1s', 2000: '2s'},
                   tooltip={'placement': 'bottom', 'always_visible': True}),
                 style={'width': '300px'}),
    ]),

    dcc.Graph(id='radar-graph',
              figure=build_figure(),
              style={'height': '75vh'},
              config={'displayModeBar': True}),

    html.Div(id='stats-bar',
             style={'marginTop': '10px', 'padding': '10px',
                    'border': '1px solid #00aa33', 'borderRadius': '4px',
                    'backgroundColor': '#050f05', 'fontSize': '13px',
                    'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap'}),

    dcc.Interval(id='interval', interval=SimulationConfig.FRAME_INTERVAL_MS, n_intervals=0),
    dcc.Store(id='store', data={'running': True}),
])


@app.callback(
    Output('store', 'data'),
    Output('btn-pause', 'children'),
    Input('btn-pause', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    State('store', 'data'),
    prevent_initial_call=True
)
def handle_buttons(_p, _r, store):
    global area, frame_count, radar_sweep_angle
    tid = callback_context.triggered_id
    if tid == 'btn-reset':
        area = RadarSimulationArea()
        frame_count = 0
        radar_sweep_angle = float(ANGLE_START)
        store['running'] = True
        return store, '⏸  Pause'
    if tid == 'btn-pause':
        store['running'] = not store['running']
    label = '▶  Resume' if not store['running'] else '⏸  Pause'
    return store, label


@app.callback(
    Output('interval', 'interval'),
    Input('speed-slider', 'value')
)
def set_speed(val):
    return val


@app.callback(
    Output('radar-graph', 'figure'),
    Output('stats-bar', 'children'),
    Input('interval', 'n_intervals'),
    State('store', 'data')
)
def tick(_, store):
    global frame_count, radar_sweep_angle

    if store and not store.get('running', True):
        return build_figure(), make_stats()

    sweep_range = ANGLE_END - ANGLE_START
    radar_sweep_angle = ANGLE_START + (
        (radar_sweep_angle - ANGLE_START + VisualizationConfig.RADAR_SWEEP_SPEED_DEG_PER_FRAME)
        % sweep_range
    )

    spawn_targets()
    area.step()
    frame_count += 1

    return build_figure(), make_stats()


def make_stats():
    confirmed = area.tracker.get_confirmed_tracks()
    items = [
        ('Frame', str(frame_count)),
        ('Sim Time', f'{area.time:.0f}s'),
        ('Targets', str(len(area.targets))),
        ('Tracks', str(len(confirmed))),
        ('Range', f'{RANGE_KM:.0f} km'),
        ('Coverage', f'{RadarConfig.COVERAGE_ANGLE_TOTAL_DEG:.0f}\u00b0'),
        ('Pd', f'{RadarConfig.DETECTION_PROBABILITY * 100:.0f}%'),
        ('Noise', f'{RadarConfig.MEASUREMENT_NOISE_M:.0f} m'),
        ('Sweep', f'{radar_sweep_angle:.1f}\u00b0'),
    ]
    return [
        html.Span([html.Strong(k + ': '), v], style={'marginRight': '8px'})
        for k, v in items
    ]


if __name__ == '__main__':
    print('=' * 60)
    print('Saab GlobalEye Web Radar  ->  http://localhost:8050')
    print('=' * 60)
    app.run(debug=True, port=8050, host='0.0.0.0')
