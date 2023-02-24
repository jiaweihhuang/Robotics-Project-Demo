import os
import pybullet as p
import pybullet_data

ClipConfig = {
    'leopard_retarget_motion': {
        'base_path': os.path.join(pybullet_data.getDataPath(), 'data', 'motions'),
        'start_list': [
                0, 115, 260, 400, 550, 700,
                850, 1090, 1250, 1460,
                1650, 1780, 1950, 2120, 2300,
                2450, 2750, 2900, 3100, 1e10
            ],
        
    }
}