import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react';

// --- Transformer Modules ---
class MultiheadAttention {
  private attention: tf.layers.MultiHeadAttention;
  private normalize: tf.layers.LayerNormalization;

  constructor(embedDim: number, numHeads: number, dropout: number = 0.0) {
    this.attention = tf.layers.multiHeadAttention({
      headCount: numHeads,
      keyDim: Math.floor(embedDim / numHeads),
      valueDim: Math.floor(embedDim / numHeads),
      dropout: dropout
    });
    this.normalize = tf.layers.layerNormalization({ axis: -1 });
  }

  async forward(input: tf.Tensor, query: tf.Tensor, padMask?: tf.Tensor, attMask?: tf.Tensor): Promise<[tf.Tensor, tf.Tensor]> {
    // input: (V,B,E) -> (B,V,E) in PyTorch, but we'll keep TF.js convention
    const inputTransposed = input.transpose([1, 0, 2]); // (B,V,E) -> (V,B,E)
    const queryTransposed = query.transpose([1, 0, 2]); // (B,Q,E) -> (Q,B,E)
    
    const [embed, att] = this.attention.apply(
      [queryTransposed, inputTransposed, inputTransposed], 
      {
        attentionMask: attMask,
        keyPaddingMask: padMask
      }
    ) as [tf.Tensor, tf.Tensor];
    
    const normalizedEmbed = this.normalize.apply(
      tf.add(embed, queryTransposed)
    ) as tf.Tensor;
    
    const finalEmbed = normalizedEmbed.transpose([1, 0, 2]); // (Q,B,E) -> (B,Q,E)
    return [finalEmbed, att];
  }
}

class PointwiseFeedForward {
  private fwdLayer: tf.LayersModel;
  private normalize: tf.layers.LayerNormalization;

  constructor(embDim: number, fwdDim: number, dropout: number = 0.0) {
    this.fwdLayer = tf.sequential({
      layers: [
        tf.layers.dense({ units: fwdDim, inputDim: embDim }),
        tf.layers.reLU(),
        tf.layers.dropout({ rate: dropout }),
        tf.layers.dense({ units: embDim })
      ]
    });
    this.normalize = tf.layers.layerNormalization({ axis: -1 });
  }

  async forward(input: tf.Tensor): Promise<tf.Tensor> {
    const output = this.fwdLayer.apply(input) as tf.Tensor; // (B,L,E)
    const normalizedOutput = this.normalize.apply(
      tf.add(output, input)
    ) as tf.Tensor; // (B,L,E)
    return normalizedOutput;
  }
}

class TransformerLayer {
  private attention: MultiheadAttention;
  private fwdLayer: PointwiseFeedForward;

  constructor(embedDim: number, numHeads: number, fwdDim: number, dropout: number = 0.0) {
    this.attention = new MultiheadAttention(embedDim, numHeads, dropout);
    this.fwdLayer = new PointwiseFeedForward(embedDim, fwdDim, dropout);
  }

  async forward(input: tf.Tensor, padMask?: tf.Tensor, attMask?: tf.Tensor): Promise<[tf.Tensor, tf.Tensor]> {
    const [emb, att] = await this.attention.forward(input, input, padMask, attMask);
    const finalEmb = await this.fwdLayer.forward(emb);
    return [finalEmb, att];
  }
}

class Xformer {
  private tokenEmbedding: tf.layers.Embedding;
  private positEmbedding: tf.layers.Embedding;
  private transform: TransformerLayer[];
  private dropout: tf.layers.Dropout;

  constructor(
    embedDim: number,
    numHeads: number,
    fwdDim: number,
    dropout: number = 0.1,
    numLayers: number = 1,
    numTokens: number = 1,
    numPosits: number = 1
  ) {
    this.tokenEmbedding = tf.layers.embedding({
      inputDim: numTokens,
      outputDim: embedDim
    });
    this.positEmbedding = tf.layers.embedding({
      inputDim: numPosits,
      outputDim: embedDim
    });
    this.transform = Array.from({ length: numLayers }, () => 
      new TransformerLayer(embedDim, numHeads, fwdDim, dropout)
    );
    this.dropout = tf.layers.dropout({ rate: dropout });
  }

  async forward(
    tokenIndex?: tf.Tensor,
    tokenEmbed?: tf.Tensor,
    padMask?: tf.Tensor,
    padId: number = -1
  ): Promise<tf.Tensor> {
    let finalEmbed: tf.Tensor;

    if (tokenIndex != null) {
      if (padMask == null) {
        padMask = tf.equal(tokenIndex, padId); // (B,L)
      }
      
      const [batchSize, seqLen] = tokenIndex.shape;
      const positIndex = tf.range(0, seqLen).expandDims(0).tile([batchSize, 1]); // (B,L)
      
      const positEmbed = this.positEmbedding.apply(positIndex) as tf.Tensor; // (B,L,E)
      const tokenEmbed = this.tokenEmbedding.apply(tokenIndex) as tf.Tensor; // (B,L,E)
      
      finalEmbed = this.dropout.apply(
        tf.add(tokenEmbed, positEmbed)
      ) as tf.Tensor; // (B,L,E)
    } else if (tokenEmbed != null) {
      const [batchSize, seqLen] = tokenEmbed.shape;
      const positIndex = tf.range(0, seqLen).expandDims(0).tile([batchSize, 1]);
      const positEmbed = this.positEmbedding.apply(positIndex) as tf.Tensor;
      finalEmbed = this.dropout.apply(
        tf.add(tokenEmbed, positEmbed)
      ) as tf.Tensor;
    } else {
      throw new Error('tokenIndex or tokenEmbed must not be null');
    }

    for (const layer of this.transform) {
      finalEmbed = (await layer.forward(finalEmbed, padMask))[0];
    }

    return finalEmbed; // (B,L,E)
  }
}

// --- Main Classifier Component ---
interface ClassifierProps {
  numTopics: number;
  numStates: number;
  vit?: any;
  xformer?: Xformer;
  fcFeatures?: number;
  embedDim?: number;
  numHeads?: number;
  dropout?: number;
}

class Classifier {
  private vit: any;
  private xformer: Xformer;
  private vitFeatures: tf.layers.Dense | null;
  private txtFeatures: MultiheadAttention | null;
  private topicEmbedding: tf.layers.Embedding;
  private stateEmbedding: tf.layers.Embedding;
  private attention: MultiheadAttention;
  private dropout: tf.layers.Dropout;
  private normalize: tf.layers.LayerNormalization;

  private numTopics: number;
  private numStates: number;

  constructor(props: ClassifierProps) {
    const {
      numTopics,
      numStates,
      vit = null,
      xformer = null,
      fcFeatures = 768,
      embedDim = 128,
      numHeads = 1,
      dropout = 0.5
    } = props;

    this.vit = vit;
    this.xformer = xformer;
    this.vitFeatures = vit != null ? 
      tf.layers.dense({ units: numTopics * embedDim, inputDim: fcFeatures }) : null;
    this.txtFeatures = xformer != null ? 
      new MultiheadAttention(embedDim, numHeads, dropout) : null;

    this.topicEmbedding = tf.layers.embedding({
      inputDim: numTopics,
      outputDim: embedDim
    });
    this.stateEmbedding = tf.layers.embedding({
      inputDim: numStates,
      outputDim: embedDim
    });
    this.attention = new MultiheadAttention(embedDim, numHeads);
    this.dropout = tf.layers.dropout({ rate: dropout });
    this.normalize = tf.layers.layerNormalization({ axis: -1 });

    this.numTopics = numTopics;
    this.numStates = numStates;
  }

  async forward(
    img?: tf.Tensor,
    txt?: tf.Tensor,
    lbl?: tf.Tensor,
    txtEmbed?: tf.Tensor,
    padMask?: tf.Tensor,
    padId: number = 3,
    threshold: number = 0.5,
    getEmbed: boolean = false,
    getTxtAtt: boolean = false
  ): Promise<tf.Tensor | [tf.Tensor, tf.Tensor]> {
    let vitFeatures: tf.Tensor | undefined;
    let txtFeatures: tf.Tensor | undefined;

    // Get vision and text features
    if (img != null) {
      vitFeatures = this.vit.forward(img); // (B,F)
    }

    if (txt != null) {
      if (padId >= 0 && padMask == null) {
        padMask = tf.equal(txt, padId);
      }
      txtFeatures = await this.xformer.forward(txt, undefined, padMask);
    } else if (txtEmbed != null) {
      txtFeatures = await this.xformer.forward(undefined, txtEmbed, padMask);
    }

    // Fuse vision and text features
    const batchSize = vitFeatures?.shape[0] || txtFeatures?.shape[0];
    if (!batchSize) throw new Error('Cannot determine batch size');

    const topicIndex = tf.range(0, this.numTopics).expandDims(0).tile([batchSize, 1]); // (B,T)
    const stateIndex = tf.range(0, this.numStates).expandDims(0).tile([batchSize, 1]); // (B,C)
    
    const topicEmbed = this.topicEmbedding.apply(topicIndex) as tf.Tensor; // (B,T,E)
    const stateEmbed = this.stateEmbedding.apply(stateIndex) as tf.Tensor; // (B,C,E)

    let finalEmbed: tf.Tensor;

    if (img != null && (txt != null || txtEmbed != null) && vitFeatures && txtFeatures) {
      const vitFeaturesReshaped = this.vitFeatures!.apply(vitFeatures) as tf.Tensor;
      const vitFeaturesReshaped2 = vitFeaturesReshaped.reshape([batchSize, this.numTopics, -1]);
      
      const [txtFeaturesAtt, txtAttention] = await this.txtFeatures!.forward(
        txtFeatures, topicEmbed, padMask
      ); // (B,T,E), (B,T,L)
      
      finalEmbed = this.normalize.apply(
        tf.add(vitFeaturesReshaped2, txtFeaturesAtt)
      ) as tf.Tensor; // (B,T,E)
    } else if (img != null && vitFeatures) {
      const vitFeaturesReshaped = this.vitFeatures!.apply(vitFeatures) as tf.Tensor;
      finalEmbed = vitFeaturesReshaped.reshape([batchSize, this.numTopics, -1]); // (B,T,E)
    } else if ((txt != null || txtEmbed != null) && txtFeatures) {
      const [txtFeaturesAtt, txtAttention] = await this.txtFeatures!.forward(
        txtFeatures, topicEmbed, padMask
      ); // (B,T,E), (B,T,L)
      finalEmbed = txtFeaturesAtt; // (B,T,E)
    } else {
      throw new Error('vision and text must not be all null');
    }

    // Classifier output
    const [emb, att] = await this.attention.forward(stateEmbed, finalEmbed); // (B,T,E), (B,T,C)

    if (lbl != null) {
      // Teacher forcing
      emb.dispose();
      const lblEmbed = this.stateEmbedding.apply(lbl) as tf.Tensor; // (B,T,E)
      if (getEmbed) {
        return [att, tf.add(finalEmbed, lblEmbed)]; // (B,T,C), (B,T,E)
      }
    } else {
      // Use threshold for inference
      const thresholdMask = tf.greater(att.slice([0, 0, 1], [batchSize, this.numTopics, 1]), threshold);
      const lblPred = tf.cast(thresholdMask, 'int32').squeeze([2]);
      const lblEmbed = this.stateEmbedding.apply(lblPred) as tf.Tensor;
      
      if (getEmbed) {
        return [att, tf.add(finalEmbed, lblEmbed)]; // (B,T,C), (B,T,E)
      }
    }

    return att; // (B,T,C)
  }
}

// --- React Component Wrapper ---
interface TransformerModelProps {
  modelConfig: {
    numTopics: number;
    numStates: number;
    embedDim: number;
    numHeads: number;
    dropout: number;
  };
  onModelReady?: (model: Classifier) => void;
}

const TransformerModel: React.FC<TransformerModelProps> = ({ 
  modelConfig, 
  onModelReady 
}) => {
  const modelRef = useRef<Classifier | null>(null);

  useEffect(() => {
    const initializeModel = async () => {
      try {
        const classifier = new Classifier(modelConfig);
        modelRef.current = classifier;
        
        if (onModelReady) {
          onModelReady(classifier);
        }
      } catch (error) {
        console.error('Failed to initialize model:', error);
      }
    };

    initializeModel();

    return () => {
      // Cleanup
      if (modelRef.current) {
        // TensorFlow.js automatically handles memory management
        // but we can explicitly dispose if needed
        tf.disposeVariables();
      }
    };
  }, [modelConfig, onModelReady]);

  return (
    <div className="transformer-model">
      <h3>Transformer Classifier Model</h3>
      <p>Model initialized with:</p>
      <ul>
        <li>Topics: {modelConfig.numTopics}</li>
        <li>States: {modelConfig.numStates}</li>
        <li>Embedding Dim: {modelConfig.embedDim}</li>
        <li>Heads: {modelConfig.numHeads}</li>
      </ul>
    </div>
  );
};

export { 
  MultiheadAttention, 
  PointwiseFeedForward, 
  TransformerLayer, 
  Xformer, 
  Classifier, 
  TransformerModel 
};
export default TransformerModel;