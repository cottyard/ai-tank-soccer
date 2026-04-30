import { FIELD, type GameState, type Tank, type Team, type Vec2 } from '../game/model';
import type { CommandMap } from '../game/strategy';

type Transform = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

export type RenderMeta = {
  redName: string;
  blueName: string;
  commands: CommandMap;
  showTrails: boolean;
};

type TrailPoint = {
  position: Vec2;
  frame: number;
};

const TEAM_COLORS: Record<Team, string> = {
  red: '#e5484d',
  blue: '#3b82f6'
};

const TEAM_DARK: Record<Team, string> = {
  red: '#7f1d1d',
  blue: '#1e3a8a'
};

export class FieldRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly ballTrail: TrailPoint[] = [];
  private transform: Transform = { scale: 1, offsetX: 0, offsetY: 0 };

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D context is unavailable');
    }
    this.ctx = ctx;
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.transform = makeTransform(rect.width, rect.height);
  }

  render(state: GameState, meta: RenderMeta): void {
    const rect = this.canvas.getBoundingClientRect();
    this.ctx.clearRect(0, 0, rect.width, rect.height);
    this.drawBackground(rect.width, rect.height);
    this.drawField();

    this.ballTrail.push({ position: { ...state.ball.position }, frame: state.frame });
    while (this.ballTrail.length > 90 || this.ballTrail[0]?.frame < state.frame - 90) {
      this.ballTrail.shift();
    }

    if (meta.showTrails) {
      this.drawBallTrail(state.frame);
    }

    this.drawGoalLabels(meta);
    for (const tank of state.tanks) {
      this.drawTank(tank, meta.commands[tank.id], state.frame);
    }
    this.drawBall(state.ball.position, state.ball.radius);
  }

  private drawBackground(width: number, height: number): void {
    const gradient = this.ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#0f1d18');
    gradient.addColorStop(0.48, '#152821');
    gradient.addColorStop(1, '#111827');
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, width, height);
  }

  private drawField(): void {
    const ctx = this.ctx;
    const topLeft = this.toScreen({ x: 0, y: 0 });
    const bottomRight = this.toScreen({ x: FIELD.length, y: FIELD.width });
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;

    ctx.save();
    ctx.translate(topLeft.x, topLeft.y);
    ctx.fillStyle = '#17603f';
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < 8; i += 1) {
      ctx.fillStyle = i % 2 === 0 ? 'rgba(255,255,255,0.035)' : 'rgba(0,0,0,0.035)';
      ctx.fillRect((width / 8) * i, 0, width / 8, height);
    }

    ctx.strokeStyle = 'rgba(240, 253, 244, 0.86)';
    ctx.lineWidth = Math.max(1.5, this.transform.scale * 3);
    ctx.strokeRect(0, 0, width, height);

    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    this.strokeCircle(FIELD.length / 2, FIELD.width / 2, 92);
    this.strokeCircle(FIELD.length / 2, FIELD.width / 2, 5);
    this.strokeBox(0, FIELD.width / 2 - 165, 165, 330);
    this.strokeBox(FIELD.length - 165, FIELD.width / 2 - 165, 165, 330);
    this.strokeBox(0, FIELD.width / 2 - 52, 52, 104);
    this.strokeBox(FIELD.length - 52, FIELD.width / 2 - 52, 52, 104);

    const wall = Math.max(5, this.transform.scale * 11);
    ctx.strokeStyle = '#273549';
    ctx.lineWidth = wall;
    ctx.strokeRect(0, 0, width, height);

    const goalTop = this.toScreen({ x: 0, y: FIELD.width / 2 - FIELD.goalMouth / 2 }).y - topLeft.y;
    const goalHeight = FIELD.goalMouth * this.transform.scale;
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = Math.max(3, this.transform.scale * 7);
    ctx.beginPath();
    ctx.moveTo(0, goalTop);
    ctx.lineTo(0, goalTop + goalHeight);
    ctx.moveTo(width, goalTop);
    ctx.lineTo(width, goalTop + goalHeight);
    ctx.stroke();

    ctx.restore();
  }

  private drawGoalLabels(meta: RenderMeta): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(255,255,255,0.72)';
    const left = this.toScreen({ x: 28, y: FIELD.width / 2 });
    const right = this.toScreen({ x: FIELD.length - 28, y: FIELD.width / 2 });
    ctx.translate(left.x, left.y);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(`${meta.blueName} target`, 0, 0);
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    ctx.translate(right.x, right.y);
    ctx.rotate(Math.PI / 2);
    ctx.fillText(`${meta.redName} target`, 0, 0);
    ctx.restore();
  }

  private drawTank(
    tank: Tank,
    command: { leftTrack: number; rightTrack: number } | undefined,
    frame: number
  ): void {
    const ctx = this.ctx;
    const center = this.toScreen(tank.position);

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(tank.angle);

    const bodyLength = tank.length * this.transform.scale;
    const bodyWidth = tank.width * this.transform.scale;
    const trackThickness = Math.max(8, bodyWidth * 0.14);

    const noseLength = tank.noseLength * this.transform.scale;
    const outline = tankShapePoints(bodyLength, bodyWidth, noseLength);

    ctx.fillStyle = 'rgba(0,0,0,0.28)';
    drawPolygon(ctx, outline.map((point) => ({ x: point.x + 2, y: point.y + 3 })));
    ctx.fill();

    ctx.fillStyle = TEAM_DARK[tank.team];
    drawPolygon(ctx, outline);
    ctx.fill();

    const inset = 0.68;
    ctx.fillStyle = TEAM_COLORS[tank.team];
    drawPolygon(ctx, outline.map((point) => ({ x: point.x * inset, y: point.y * inset })));
    ctx.fill();

    const leftPower = command?.leftTrack ?? 0;
    const rightPower = command?.rightTrack ?? 0;
    this.drawTrack(-bodyWidth * 0.42, bodyLength, trackThickness, leftPower, frame);
    this.drawTrack(bodyWidth * 0.42, bodyLength, trackThickness, rightPower, frame);
    this.drawStaminaBar(tank, bodyLength, bodyWidth);
    ctx.restore();
  }

  private drawTrack(
    y: number,
    bodyLength: number,
    thickness: number,
    power: number,
    phase: number
  ): void {
    const ctx = this.ctx;
    const x = -bodyLength / 2;
    const patternWidth = Math.max(9, bodyLength * 0.14);
    const stripeWidth = Math.max(4, patternWidth * 0.45);
    const speed = patternWidth * 0.075;
    const offset = power === 0 ? 0 : positiveModulo(phase * speed * power, patternWidth);

    ctx.fillStyle = '#111827';
    roundRect(ctx, -bodyLength / 2, y - thickness / 2, bodyLength, thickness, 4);
    ctx.fill();

    ctx.save();
    roundRect(ctx, x, y - thickness / 2, bodyLength, thickness, 4);
    ctx.clip();
    ctx.fillStyle = power === 0 ? 'rgba(148, 163, 184, 0.28)' : 'rgba(226, 232, 240, 0.78)';
    for (let stripeX = x - patternWidth + offset; stripeX < bodyLength / 2 + patternWidth; stripeX += patternWidth) {
      ctx.save();
      ctx.translate(stripeX + stripeWidth / 2, y);
      ctx.rotate(-0.42);
      ctx.fillRect(-stripeWidth / 2, -thickness, stripeWidth, thickness * 2);
      ctx.restore();
    }
    ctx.restore();

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.78)';
    ctx.lineWidth = Math.max(1, thickness * 0.13);
    roundRect(ctx, x, y - thickness / 2, bodyLength, thickness, 4);
    ctx.stroke();
  }

  private drawStaminaBar(tank: Tank, bodyLength: number, bodyWidth: number): void {
    const ctx = this.ctx;
    const width = bodyLength * 0.74;
    const height = Math.max(4, bodyWidth * 0.06);
    const x = -width / 2;
    const y = -bodyWidth * 0.5 - height - 3;
    const ratio = Math.max(0, Math.min(1, tank.stamina / tank.maxStamina));

    ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
    roundRect(ctx, x, y, width, height, 3);
    ctx.fill();
    ctx.fillStyle = ratio > 0.35 ? '#bef264' : '#fbbf24';
    roundRect(ctx, x, y, width * ratio, height, 3);
    ctx.fill();
  }

  private drawBall(position: Vec2, radius: number): void {
    const ctx = this.ctx;
    const center = this.toScreen(position);
    const screenRadius = radius * this.transform.scale;
    const gradient = ctx.createRadialGradient(
      center.x - screenRadius * 0.35,
      center.y - screenRadius * 0.35,
      screenRadius * 0.1,
      center.x,
      center.y,
      screenRadius
    );
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.45, '#e2e8f0');
    gradient.addColorStop(1, '#64748b');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(center.x, center.y, screenRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = Math.max(1.5, screenRadius * 0.08);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(15,23,42,0.45)';
    ctx.lineWidth = Math.max(1, screenRadius * 0.045);
    ctx.beginPath();
    ctx.arc(center.x, center.y, screenRadius * 0.56, Math.PI * 0.08, Math.PI * 1.42);
    ctx.stroke();
  }

  private drawBallTrail(currentFrame: number): void {
    const ctx = this.ctx;
    for (let i = 1; i < this.ballTrail.length; i += 1) {
      const previous = this.ballTrail[i - 1];
      const point = this.ballTrail[i];
      const age = Math.min(1, (currentFrame - point.frame) / 90);
      const a = this.toScreen(previous.position);
      const b = this.toScreen(point.position);
      ctx.strokeStyle = `rgba(250, 204, 21, ${0.35 * (1 - age)})`;
      ctx.lineWidth = Math.max(1, this.transform.scale * FIELD.ballRadius * (0.16 - age * 0.08));
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  private strokeCircle(x: number, y: number, radius: number): void {
    const ctx = this.ctx;
    const center = this.toLocalScreen({ x, y });
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius * this.transform.scale, 0, Math.PI * 2);
    ctx.stroke();
  }

  private strokeBox(x: number, y: number, width: number, height: number): void {
    const ctx = this.ctx;
    const point = this.toLocalScreen({ x, y });
    ctx.strokeRect(point.x, point.y, width * this.transform.scale, height * this.transform.scale);
  }

  private toScreen(point: Vec2): Vec2 {
    return {
      x: this.transform.offsetX + point.x * this.transform.scale,
      y: this.transform.offsetY + point.y * this.transform.scale
    };
  }

  private toLocalScreen(point: Vec2): Vec2 {
    return {
      x: point.x * this.transform.scale,
      y: point.y * this.transform.scale
    };
  }
}

export function makeTransform(width: number, height: number): Transform {
  const padding = Math.max(16, Math.min(width, height) * 0.035);
  const scale = Math.min((width - padding * 2) / FIELD.length, (height - padding * 2) / FIELD.width);
  const fieldWidth = FIELD.length * scale;
  const fieldHeight = FIELD.width * scale;
  return {
    scale,
    offsetX: (width - fieldWidth) / 2,
    offsetY: (height - fieldHeight) / 2
  };
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function positiveModulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function tankShapePoints(bodyLength: number, bodyWidth: number, noseLength: number): Vec2[] {
  const halfLength = bodyLength / 2;
  const halfWidth = bodyWidth / 2;

  return [
    { x: -halfLength, y: -halfWidth },
    { x: halfLength, y: -halfWidth },
    { x: halfLength + noseLength, y: -halfWidth },
    { x: halfLength, y: 0 },
    { x: halfLength + noseLength, y: halfWidth },
    { x: halfLength, y: halfWidth },
    { x: -halfLength, y: halfWidth }
  ];
}

function drawPolygon(ctx: CanvasRenderingContext2D, points: Vec2[]): void {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) {
    ctx.lineTo(points[index].x, points[index].y);
  }
  ctx.closePath();
}
