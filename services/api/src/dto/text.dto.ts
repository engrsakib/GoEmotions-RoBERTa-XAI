import { IsNotEmpty, IsString, MaxLength, MinLength } from 'class-validator';

export class TextDto {
  @IsString()
  @IsNotEmpty()
  @MinLength(1)
  @MaxLength(2000)
  text!: string;
}

export interface PredictResponse {
  category: number;
  label: string;
  display_label: string;
  confidence: number;
  scores: Record<string, number>;
}

export interface ExplainResponse extends PredictResponse {
  tokens: string[];
  heatmap: number[];
  method: string;
}

export interface ChatResponse extends ExplainResponse {
  reply: string;
}

export interface HealthResponse {
  status: string;
  gateway: string;
  model: {
    status: string;
    model_loaded: boolean;
    model_path: string;
    device: string;
  };
}
