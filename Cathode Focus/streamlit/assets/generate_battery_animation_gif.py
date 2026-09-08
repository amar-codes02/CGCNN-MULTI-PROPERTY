import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def create_battery_animation():
    fig = plt.figure(figsize=(10, 5.5), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0f172a')
    
    frames = []
    soc_steps = list(np.linspace(100, 0, 25)) + list(np.linspace(0, 100, 25))
    
    np.random.seed(42)
    
    # Static TPMS Host Node Coordinates
    u = np.linspace(3, 11, 8)
    v = np.linspace(-5, 5, 8)
    w = np.linspace(-5, 5, 8)
    gx, gy, gz = np.meshgrid(u, v, w)
    tpms_val = np.sin(gx*0.8)*np.cos(gy*0.8) + np.sin(gy*0.8)*np.cos(gz*0.8) + np.sin(gz*0.8)*np.cos(gx*0.8)
    mask = np.abs(tpms_val) < 0.65
    cx, cy, cz = gx[mask], gy[mask], gz[mask]

    for step_i, soc in enumerate(soc_steps):
        ax.clear()
        ax.set_facecolor('#0f172a')
        
        is_discharge = step_i < 25
        mode_str = "DISCHARGING (Pengosongan)" if is_discharge else "CHARGING (Pengisian Daya)"
        
        # Voltage profile simulation
        if is_discharge:
            if soc >= 80:
                v_cell = 2.40 - (100 - soc) * 0.005
                stage_str = "Stage I: S8 -> Li2S8 (Soluble)"
                color_poly = "#facc15"
            elif soc >= 60:
                v_cell = 2.30 - (80 - soc) * 0.0075
                stage_str = "Stage II: Li2S8 -> Li2S6 -> Li2S4"
                color_poly = "#fb923c"
            elif soc >= 20:
                v_cell = 2.10 - (60 - soc) * 0.0025
                stage_str = "Stage III: Li2S4 -> Li2S2 (Precipitation)"
                color_poly = "#a855f7"
            else:
                v_cell = 2.05 - (20 - soc) * 0.0175
                stage_str = "Stage IV: Li2S2 -> Li2S Solid Layer"
                color_poly = "#ef4444"
        else:
            v_cell = 2.15 + (soc / 100.0) * 0.30
            stage_str = "Charging Oxidation: Li2S -> Li2Sx -> S8"
            color_poly = "#38bdf8"
            
        # 1. Anode Li Metal Grid (x in [-12, -5])
        n_anode_atoms = int(15 + (soc / 100.0) * 20) if is_discharge else int(15 + ((100-soc) / 100.0) * 20)
        ax.scatter([-10]*n_anode_atoms, np.random.uniform(-5, 5, n_anode_atoms), np.random.uniform(-5, 5, n_anode_atoms),
                   color='#94a3b8', s=45, label='Anode Li Metal (Li⁰)', depthshade=True)

        # 2. Migrating Li+ Ions (x in [-5, 3])
        ion_dir = -1 if is_discharge else 1
        ion_x = np.random.uniform(-5, 2, 22) + ion_dir * (step_i % 5) * 0.3
        ax.scatter(ion_x, np.random.uniform(-5, 5, 22), np.random.uniform(-5, 5, 22),
                   color='#facc15', s=35, marker='^', label='Li⁺ Ions Migrating')

        # 3. Cathode Graphene TPMS Host
        ax.scatter(cx, cy, cz, color='#475569', s=30, alpha=0.6, label='Graphene TPMS Host')

        # 4. Anchored Polysulfides on TPMS Surface
        n_poly = int((100 - soc) / 100.0 * len(cx) * 0.35)
        if n_poly > 0:
            poly_idx = np.random.choice(len(cx), min(n_poly, len(cx)), replace=False)
            ax.scatter(cx[poly_idx], cy[poly_idx], cz[poly_idx] + 0.3, color=color_poly, s=65, label=f'Polysulfide ({stage_str.split(":")[1].strip()})')

        # 5. External Circuit Electron Path
        circuit_x = np.linspace(-10, 8, 15)
        circuit_z = 7.5 + np.sin(np.linspace(0, np.pi, 15)) * 1.5
        ax.plot(circuit_x, [0]*15, circuit_z, color='#38bdf8', linewidth=2.5, linestyle='--')

        # Title & Annotations
        title_text = f"Li-S Battery 3D Dynamic Cycle | Mode: {mode_str}\nSOC: {soc:.0f}% | Voltage: {v_cell:.2f} V | {stage_str}"
        ax.set_title(title_text, color='#f8fafc', fontsize=11, fontweight='bold', pad=12)
        
        ax.set_xlim(-12, 12)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 10)
        ax.axis('off')
        
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        frames.append(rgb)

    plt.close(fig)
    gif_path = 'lis_battery_redox_cycle_animation.gif'
    imageio.mimsave(gif_path, frames, fps=6)
    print(f" Animation GIF created successfully: {gif_path}")

if __name__ == '__main__':
    create_battery_animation()
