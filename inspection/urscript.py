from .hardware import POSE_CODES, StationConfig


def encode_angles(joints):
    return '[' + ', '.join(f'{angle:.9f}' for angle in joints) + ']'


def build_script(station, output):
    config = StationConfig(station)
    home = encode_angles(config.home)
    goals = '\n'.join(f'    elif code == {code}:\n      return {encode_angles(route[-1])}'
                      for code, route in config.routes.items())
    messages = '\n'.join(f'    elif code == {code}:\n      return socket_send_line("AT {code}", "qc")'
                         for code in POSE_CODES.values())
    routes = []
    for code, route in config.routes.items():
        branch = 'if' if code == 1 else 'elif'
        routes.append(f'    {branch} code == {code}:\n      if leaving:')
        for waypoint in [*reversed(route[:-1]), config.home]:
            routes.append(f'        qc_move({encode_angles(waypoint)})')
        routes.append('      else:')
        for waypoint in route:
            routes.append(f'        qc_move({encode_angles(waypoint)})')
        routes.append('      end')
    branches = '\n'.join(routes)
    script = f'''# UR5e fixed-fixture inspection. Load as a PolyScope 5 Script file node.
def qc_station():
  def qc_goal(code):
    if code == 0:
      return {home}
{goals}
    end
  end

  def qc_at(goal):
    local actual = get_actual_joint_positions()
    local speeds = get_actual_joint_speeds()
    local joint = 0
    while joint < 6:
      if abs(actual[joint] - goal[joint]) > 0.015 or abs(speeds[joint]) > 0.02:
        return False
      end
      joint = joint + 1
    end
    return True
  end

  def qc_abort():
    socket_close("qc")
    textmsg("Screw inspection stopped. Check station and recover to home manually.")
    halt
  end

  def qc_move(joints):
    movej(joints, a={config.acceleration:.6f}, v={config.speed:.6f}, r=0)
    sleep(0.25)
    if not qc_at(joints):
      qc_abort()
    end
  end

  def qc_route(code, leaving):
{branches}
    end
  end

  def qc_arrived(code):
    if code == 0:
      return socket_send_line("AT 0", "qc")
{messages}
    end
  end

  while True:
    if not qc_at(qc_goal(0)):
      textmsg("Start the screw inspection program at the taught home position.")
      halt
    end
    if socket_open("{config.pc_ip}", {config.port}, "qc"):
      if not socket_send_line("READY {config.route_token}", "qc"):
        qc_abort()
      end
      local current = 0
      local completed = False
      while not completed:
        local command = socket_read_ascii_float(1, "qc", 30.0)
        if command[0] != 1:
          qc_abort()
        end
        local target = command[1]
        if target < 0 or target > {len(POSE_CODES)} or floor(target) != target:
          qc_abort()
        end
        if not qc_at(qc_goal(current)):
          qc_abort()
        end
        if current != 0:
          qc_route(current, True)
        end
        if target != 0:
          qc_route(target, False)
        end
        current = target
        if not qc_arrived(current):
          qc_abort()
        end
        completed = current == 0
      end
      socket_close("qc")
    end
    sleep(0.5)
  end
end

qc_station()
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(script)
